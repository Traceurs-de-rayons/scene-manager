#include "sceneDescriptor.hpp"
#include "Scene.hpp"
#include "Vertex.hpp"
#include "renderApi.hpp"
#include "buffer.hpp"
#include "image.hpp"
#include "return.hpp"
#include "stb_image.h"

#include <iostream>
#include <vector>
#include <vulkan/vulkan_core.h>

SceneDescriptor::SceneDescriptor(std::string name, Scene* scene, renderApi::device::GPU* gpu)
		: _name(std::move(name)), _scene(scene) {
	_resources.gpu = gpu;
}

SceneDescriptor::~SceneDescriptor() {
	unloadScene();
}

Result SceneDescriptor::loadScene() {
	if (_isLoaded)
		return Result::ok();

	if (!_resources.gpu)
		return Result::error("SceneDescriptor '" + _name + "': no GPU set");

	if (!_scene)
		return Result::error("SceneDescriptor '" + _name + "': no scene set");

	if (_scene->getVertexAmount() == 0 || _scene->getIndexAmount() == 0) {
		std::cout << "SceneDescriptor '" << _name << "': empty scene, loading fallback pyramid" << std::endl;
		return loadFallbackPyramid();
	}

	return loadFromScene();
}


static std::string materialTextureName(const Material& material) {
	if (material.albedo && std::holds_alternative<std::string>(material.albedo->value))
		return std::get<std::string>(material.albedo->value);
	return "";
}

// The shaders (RowMajor, mul(M, v)) expect model matrices transposed relative to
// cu::math's column-vector convention used by Transform::getMatrix().
static mat4 transposeMatrix(const mat4& m) {
	mat4 r(0.0f);
	for (int row = 0; row < 4; ++row)
		for (int col = 0; col < 4; ++col)
			r.m[col][row] = m.m[row][col];
	return r;
}

Result SceneDescriptor::loadFromScene() {
	_resources.vertexCount = _scene->getVertexAmount();
	_resources.indexCount = _scene->getIndexAmount();

	_resources.vertexBuffer = renderApi::createVertexBuffer<Vertex>(_resources.gpu, _resources.vertexCount);
	if (!_resources.vertexBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create vertex buffer");

	_resources.indexBuffer = renderApi::createIndexBuffer<uint32_t>(_resources.gpu, _resources.indexCount);
	if (!_resources.indexBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create index buffer");

	{
		std::vector<Vertex> vertices(_resources.vertexCount);
		_scene->loadVertices(vertices.data());
		if (!_resources.vertexBuffer.upload(vertices.data(), vertices.size() * sizeof(Vertex)))
			return Result::error("SceneDescriptor '" + _name + "': failed to upload vertices");
	}

	{
		std::vector<uint32_t> indices(_resources.indexCount);
		_scene->loadIndices(indices.data());
		if (!_resources.indexBuffer.upload(indices.data(), indices.size() * sizeof(uint32_t)))
			return Result::error("SceneDescriptor '" + _name + "': failed to upload indices");
	}


	std::vector<VkDrawIndexedIndirectCommand> drawCommands;
	std::vector<mat4> matrices;
	std::vector<uint32_t> perDrawMaterialIds;
	std::vector<std::string> materialTextureNames;
	std::unordered_map<std::string, uint32_t> materialIds;

	_scene->forEachSubMesh([&](const DrawCallView& drawCall) {
		matrices.push_back(transposeMatrix(drawCall.transform.getMatrix()));

		auto it = materialIds.find(drawCall.material.name);
		uint32_t materialId;
		if (it == materialIds.end()) {
			materialId = static_cast<uint32_t>(materialTextureNames.size());
			materialIds[drawCall.material.name] = materialId;
			materialTextureNames.push_back(materialTextureName(drawCall.material));
		} else {
			materialId = it->second;
		}
		perDrawMaterialIds.push_back(materialId);

		VkDrawIndexedIndirectCommand cmd{};
		cmd.indexCount    = drawCall.indecesCount;
		cmd.instanceCount = 1;
		cmd.firstIndex    = drawCall.indicesIndex;
		cmd.vertexOffset  = static_cast<int32_t>(drawCall.verticesIndex);
		cmd.firstInstance = 0;
		drawCommands.push_back(cmd);
	});

	if (matrices.empty())
		matrices.push_back(mat4::identity());

	_resources.modelMatrixBuffer = renderApi::createStorageBuffer(_resources.gpu, matrices, renderApi::BufferUsage::DYNAMIC);
	if (!_resources.modelMatrixBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create model matrix buffer");

	_resources.drawCount = static_cast<uint32_t>(drawCommands.size());
	_resources.drawCommandsBuffer = renderApi::createIndirectBuffer(
		_resources.gpu, std::max<size_t>(1, drawCommands.size() * sizeof(VkDrawIndexedIndirectCommand)));
	if (!_resources.drawCommandsBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create draw commands buffer");
	if (!drawCommands.empty() && !_resources.drawCommandsBuffer.upload(
			drawCommands.data(), drawCommands.size() * sizeof(VkDrawIndexedIndirectCommand)))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload draw commands");

	_resources.perDrawBuffer = renderApi::createStorageBuffer(
		_resources.gpu, std::max<size_t>(1, perDrawMaterialIds.size() * sizeof(uint32_t)));
	if (!_resources.perDrawBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create per-draw buffer");
	if (!perDrawMaterialIds.empty() && !_resources.perDrawBuffer.upload(
			perDrawMaterialIds.data(), perDrawMaterialIds.size() * sizeof(uint32_t)))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload per-draw data");


	loadTextures();
	if (_resources.textures.empty())
		return ensureFallbackTexture();


	std::vector<uint32_t> materialData;
	materialData.reserve(materialTextureNames.size());
	for (const std::string& texName : materialTextureNames) {
		uint32_t textureIndex = 0;
		auto it = _textureIndexMap.find(texName);
		if (!texName.empty() && it != _textureIndexMap.end())
			textureIndex = it->second;
		materialData.push_back(textureIndex);
	}
	if (materialData.empty())
		materialData.push_back(0);

	_resources.materialBuffer = renderApi::createStorageBuffer(
		_resources.gpu, materialData.size() * sizeof(uint32_t));
	if (!_resources.materialBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create material buffer");
	if (!_resources.materialBuffer.upload(materialData.data(), materialData.size() * sizeof(uint32_t)))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload material data");

	_scene->loadHierarchy();

	_isLoaded = true;
	std::cout << "SceneDescriptor '" << _name << "': loaded ("
		  << _resources.vertexCount << " vertices, "
		  << _resources.indexCount << " indices, "
		  << _resources.drawCount << " draws, "
		  << materialData.size() << " materials, "
		  << _resources.textures.size() << " textures)" << std::endl;
	return Result::ok();
}

void SceneDescriptor::syncModelMatrices() {
	if (!_scene || !_resources.modelMatrixBuffer.isValid())
		return;

	std::vector<mat4> matrices;
	_scene->forEachSubMesh([&](const DrawCallView& drawCall) {
		matrices.push_back(transposeMatrix(drawCall.transform.getMatrix()));
	});

	if (matrices.empty())
		return;

	_resources.modelMatrixBuffer.upload(matrices.data(), matrices.size() * sizeof(mat4));
}

void SceneDescriptor::collectLights(std::vector<RasterCore::Light>& out) const {
	if (!_scene)
		return;

	// Suns first: they get the shadow map before plain directional lights.
	for (const auto& [name, asset] : _scene->getAssets()) {
		const auto* sun = asset.getType() == AssetType::Sun ? asset.getSunData() : nullptr;
		if (!sun)
			continue;

		RasterCore::Light light;
		light.type = RasterCore::LightType::Directional;
		light.direction = {sun->direction.x, sun->direction.y, sun->direction.z};
		light.color = {sun->color.x, sun->color.y, sun->color.z};
		light.intensity = sun->intensity;
		light.angularDiameter = sun->angle;
		light.castsShadow = true;
		out.push_back(light);
	}

	for (const auto& [name, asset] : _scene->getAssets()) {
		const auto* data = asset.getType() == AssetType::Light ? asset.getLightData() : nullptr;
		if (!data)
			continue;

		RasterCore::Light light;
		light.color = {data->color.x, data->color.y, data->color.z};
		light.intensity = data->intensity;
		light.castsShadow = true; // the renderer decides which ones actually get a shadow map

		// projection: 0 = Point, 1 = Directional (Asset::LightData is private)
		if (const auto* point = std::get_if<0>(&data->projection)) {
			light.type = RasterCore::LightType::Point;
			light.position = {point->position.x, point->position.y, point->position.z};
		} else if (const auto* directional = std::get_if<1>(&data->projection)) {
			light.type = RasterCore::LightType::Directional;
			light.direction = {directional->direction.x, directional->direction.y, directional->direction.z};
		}
		out.push_back(light);
	}
}

Result SceneDescriptor::loadFallbackPyramid() {
	std::vector<Vertex> vertices = {
		{{-0.5f, 0.0f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
		{{ 0.5f, 0.0f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
		{{ 0.5f, 0.0f,  0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
		{{-0.5f, 0.0f,  0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
		{{ 0.0f, 1.0f,  0.0f}, {0.0f,  1.0f, 0.0f}, {0.5f, 0.5f}},
	};

	std::vector<uint32_t> indices = {
		0, 2, 1,  0, 3, 2,
		0, 1, 4,
		1, 2, 4,
		2, 3, 4,
		3, 0, 4,
	};

	std::vector<mat4> matrices = { mat4::identity() };

	_resources.vertexCount = static_cast<uint32_t>(vertices.size());
	_resources.indexCount = static_cast<uint32_t>(indices.size());

	_resources.vertexBuffer = renderApi::createVertexBuffer<Vertex>(_resources.gpu, _resources.vertexCount);
	if (!_resources.vertexBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create vertex buffer");
	if (!_resources.vertexBuffer.upload(vertices.data(), vertices.size() * sizeof(Vertex)))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload vertices");

	_resources.indexBuffer = renderApi::createIndexBuffer<uint32_t>(_resources.gpu, _resources.indexCount);
	if (!_resources.indexBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create index buffer");
	if (!_resources.indexBuffer.upload(indices.data(), indices.size() * sizeof(uint32_t)))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload indices");

	_resources.modelMatrixBuffer = renderApi::createStorageBuffer(_resources.gpu, matrices, renderApi::BufferUsage::DYNAMIC);
	if (!_resources.modelMatrixBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create model matrix buffer");

	VkDrawIndexedIndirectCommand cmd{};
	cmd.indexCount    = _resources.indexCount;
	cmd.instanceCount = 1;
	cmd.firstIndex    = 0;
	cmd.vertexOffset  = 0;
	cmd.firstInstance = 0;

	_resources.drawCount = 1;
	_resources.drawCommandsBuffer = renderApi::createIndirectBuffer(_resources.gpu, sizeof(cmd));
	if (!_resources.drawCommandsBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create draw commands buffer");
	if (!_resources.drawCommandsBuffer.upload(&cmd, sizeof(cmd)))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload draw commands");

	const uint32_t zero = 0;
	_resources.perDrawBuffer = renderApi::createStorageBuffer(_resources.gpu, sizeof(uint32_t));
	if (!_resources.perDrawBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create per-draw buffer");
	if (!_resources.perDrawBuffer.upload(&zero, sizeof(uint32_t)))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload per-draw data");

	_resources.materialBuffer = renderApi::createStorageBuffer(_resources.gpu, sizeof(uint32_t));
	if (!_resources.materialBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create material buffer");
	if (!_resources.materialBuffer.upload(&zero, sizeof(uint32_t)))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload material data");

	Result texResult = ensureFallbackTexture();
	if (texResult.code == ResultCode::Error)
		return texResult;

	_isLoaded = true;
	std::cout << "SceneDescriptor '" << _name << "': fallback pyramid loaded" << std::endl;
	return Result::ok();
}

Result SceneDescriptor::loadTextures() {
	if (!_scene)
		return Result::ok();

	_textureIndexMap.clear();

	const auto& sceneTextures = _scene->getTextures();
	if (sceneTextures.empty())
		return Result::ok();

	for (const auto& [key, tex] : sceneTextures) {
		if (!std::holds_alternative<Texture::FromFile>(tex.data))
			continue;

		const auto& ff = std::get<Texture::FromFile>(tex.data);
		if (ff.path.empty())
			continue;

		int w = 0, h = 0, channels = 0;
		stbi_uc* pixels = stbi_load(ff.path.c_str(), &w, &h, &channels, STBI_rgb_alpha);
		if (!pixels) {
			std::cerr << "SceneDescriptor '" << _name << "': failed to load texture '"
					  << ff.path << "' (" << stbi_failure_reason() << ")" << std::endl;
			continue;
		}

		renderApi::Texture gpuTex = renderApi::createTexture2D(
			_resources.gpu,
			static_cast<uint32_t>(w),
			static_cast<uint32_t>(h),
			VK_FORMAT_R8G8B8A8_UNORM,
			pixels,
			static_cast<size_t>(w) * h * 4,
			false);

		stbi_image_free(pixels);

		if (!gpuTex.isValid()) {
			std::cerr << "SceneDescriptor '" << _name << "': failed to create GPU texture '"
					  << ff.path << "'" << std::endl;
			continue;
		}

		_textureIndexMap[key] = static_cast<uint32_t>(_resources.textures.size());
		_resources.textures.push_back(std::move(gpuTex));
		std::cout << "SceneDescriptor '" << _name << "': texture '" << tex.name
				  << "' (" << w << "x" << h << ") uploaded" << std::endl;
	}

	return Result::ok();
}

Result SceneDescriptor::ensureFallbackTexture() {
	if (!_resources.textures.empty())
		return Result::ok();

	const uint32_t whitePixel = 0xffffffffu;
	renderApi::Texture whiteTex = renderApi::createTexture2D(
		_resources.gpu, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
		&whitePixel, sizeof(whitePixel), false);
	if (!whiteTex.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create default texture");
	_resources.textures.push_back(std::move(whiteTex));
	return Result::ok();
}

Result SceneDescriptor::unloadScene() {
	if (!_isLoaded)
		return Result::ok();

	if (_resources.gpu)
		vkDeviceWaitIdle(_resources.gpu->device);

	_resources.destroy();
	_isLoaded = false;
	std::cout << "SceneDescriptor '" << _name << "': unloaded" << std::endl;
	return Result::ok();
}

void								SceneDescriptor::setName(const std::string& name) { this->_name = name; }
const std::string&					SceneDescriptor::getName() const { return _name; }
void								SceneDescriptor::setScene(Scene* scene) { this->_scene = scene; }
Scene*								SceneDescriptor::getScene() const { return _scene; }
RasterCore::SharedGpuResources*		SceneDescriptor::getResources() { return &_resources; }
const RasterCore::SharedGpuResources* SceneDescriptor::getResources() const { return &_resources; }
bool								SceneDescriptor::isLoadedScene() const { return _isLoaded; }
