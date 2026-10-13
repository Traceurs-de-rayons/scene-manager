#include <optional>
#include <algorithm>
#include "sceneDescriptor.hpp"
#include "Scene.hpp"
#include "Vertex.hpp"
#include "renderApi.hpp"
#include "buffer.hpp"
#include "image.hpp"
#include "return.hpp"
#include "stb_image.h"

#include <cstring>
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
	// New geometry: whatever was rendered from the previous one (cached shadow maps) is stale.
	++_resources.geometryVersion;
	_uploadedMatrices.clear();

	if (_isLoaded)
		return Result::ok();

	if (!_resources.gpu)
		return Result::error("SceneDescriptor '" + _name + "': no GPU set");

	if (!_scene)
		return Result::error("SceneDescriptor '" + _name + "': no scene set");

	_loadedStructureVersion = _scene->getStructureVersion();

	if (_scene->getVertexAmount() == 0 || _scene->getIndexAmount() == 0) {
		std::cout << "SceneDescriptor '" << _name << "': empty scene, loading fallback pyramid" << std::endl;
		return loadFallbackPyramid();
	}

	return loadFromScene();
}


// Size of the texture array of geometry.frag.
static constexpr size_t kMaxSceneTextures = 128;
// Largest width or height a scene texture is uploaded with.
static constexpr int kMaxTextureSize = 2048;

static std::optional<vec3> materialColor(const Material& material) {
	if (material.albedo && std::holds_alternative<cu::math::vec3>(material.albedo->value))
		return std::get<cu::math::vec3>(material.albedo->value);
	return std::nullopt;
}

// 0 = plain surface, 1 = fully emissive.
static float materialEmission(const Material& material) {
	if (material.emission_strength && std::holds_alternative<float>(material.emission_strength->value))
		return std::clamp(std::get<float>(material.emission_strength->value), 0.0f, 1.0f);
	return 0.0f;
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

// Hidden draws get a null matrix: every vertex collapses to one point, so nothing is rasterized
// (main pass and shadow maps alike) and the draw list stays untouched.
static mat4 drawCallMatrix(const DrawCallView& drawCall) {
	if (!drawCall.visible)
		return mat4(0.0f);
	return transposeMatrix(drawCall.transform.getMatrix());
}

// Bone matrices of every skinned draw, one pose after the other, in the layout the shaders
// expect. `firstBones`, when given, receives the drawInfoBuffer entry of each draw.
static std::vector<mat4> collectBones(const Scene& scene, std::vector<uint32_t>* drawInfos = nullptr) {
	std::vector<mat4> bones;
	std::unordered_map<const std::vector<mat4>*, uint32_t> firstBones; // sub meshes of one asset share its pose

	scene.forEachSubMesh([&](const DrawCallView& drawCall) {
		uint32_t first = 0;
		if (drawCall.bones) {
			auto it = firstBones.find(drawCall.bones);
			if (it == firstBones.end()) {
				first = static_cast<uint32_t>(bones.size());
				firstBones[drawCall.bones] = first;
				for (const mat4& bone : *drawCall.bones)
					bones.push_back(transposeMatrix(bone));
			} else {
				first = it->second;
			}
		}
		if (drawInfos)
			drawInfos->push_back(first | (drawCall.viewModel ? RasterCore::SharedGpuResources::kViewModelBit : 0u));
	});

	if (bones.empty())
		bones.push_back(mat4::identity());
	return bones;
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

	_resources.skinBuffer = renderApi::createVertexBuffer<VertexSkin>(_resources.gpu, _resources.vertexCount);
	if (!_resources.skinBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create skin buffer");
	{
		std::vector<VertexSkin> skins(_resources.vertexCount);
		_scene->loadVertexSkins(skins.data());
		if (!_resources.skinBuffer.upload(skins.data(), skins.size() * sizeof(VertexSkin)))
			return Result::error("SceneDescriptor '" + _name + "': failed to upload vertex skins");
	}

	{
		std::vector<uint32_t> indices(_resources.indexCount);
		_scene->loadIndices(indices.data());
		if (!_resources.indexBuffer.upload(indices.data(), indices.size() * sizeof(uint32_t)))
			return Result::error("SceneDescriptor '" + _name + "': failed to upload indices");
	}


	std::vector<VkDrawIndexedIndirectCommand>& drawCommands = _drawCommands;
	drawCommands.clear();
	_drawVisible.clear();
	std::vector<mat4> matrices;
	std::vector<uint32_t> perDrawMaterialIds;
	std::vector<std::string> materialTextureNames;
	std::vector<std::optional<vec3>> materialColors; // flat albedo of the materials that have no texture
	std::vector<float> materialEmissions;
	std::unordered_map<std::string, uint32_t>& materialIds = _materialIds;
	materialIds.clear();

	_scene->forEachSubMesh([&](const DrawCallView& drawCall) {
		matrices.push_back(drawCallMatrix(drawCall));

		auto it = materialIds.find(drawCall.material.name);
		uint32_t materialId;
		if (it == materialIds.end()) {
			materialId = static_cast<uint32_t>(materialTextureNames.size());
			materialIds[drawCall.material.name] = materialId;
			materialTextureNames.push_back(materialTextureName(drawCall.material));
			materialColors.push_back(materialColor(drawCall.material));
			materialEmissions.push_back(materialEmission(drawCall.material));
		} else {
			materialId = it->second;
		}
		perDrawMaterialIds.push_back(materialId);

		VkDrawIndexedIndirectCommand cmd{};
		cmd.indexCount    = drawCall.indecesCount;
		cmd.instanceCount = drawCall.visible ? 1 : 0; // a hidden draw costs nothing: see syncModelMatrices()
		cmd.firstIndex    = drawCall.indicesIndex;
		cmd.vertexOffset  = static_cast<int32_t>(drawCall.verticesIndex);
		cmd.firstInstance = 0;
		drawCommands.push_back(cmd);
		_drawVisible.push_back(drawCall.visible);
	});

	// Materials no draw uses yet get an id too: an asset may switch to one of them later
	// (Asset::setMaterialOverride()) without anything being rebuilt.
	for (const auto& [name, material] : _scene->getMaterials()) {
		if (materialIds.find(name) != materialIds.end())
			continue;
		materialIds[name] = static_cast<uint32_t>(materialTextureNames.size());
		materialTextureNames.push_back(materialTextureName(material));
		materialColors.push_back(materialColor(material));
		materialEmissions.push_back(materialEmission(material));
	}
	_uploadedMaterialIds = perDrawMaterialIds;

	if (matrices.empty())
		matrices.push_back(mat4::identity());

	_resources.modelMatrixBuffer = renderApi::createStorageBuffer(_resources.gpu, matrices, renderApi::BufferUsage::DYNAMIC);
	if (!_resources.modelMatrixBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create model matrix buffer");

	_resources.drawCount = static_cast<uint32_t>(drawCommands.size());
	// Showing or hiding an asset changes its instance count: see SharedGpuResources::drawCommands.
	if (!_resources.createDrawCommands(drawCommands.data(), drawCommands.size() * sizeof(VkDrawIndexedIndirectCommand)))
		return Result::error("SceneDescriptor '" + _name + "': failed to create draw commands buffer");

	_resources.perDrawBuffer = renderApi::createStorageBuffer(
		_resources.gpu, std::max<size_t>(1, perDrawMaterialIds.size() * sizeof(uint32_t)), renderApi::BufferUsage::DYNAMIC);
	if (!_resources.perDrawBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create per-draw buffer");
	if (!perDrawMaterialIds.empty() && !_resources.perDrawBuffer.upload(
			perDrawMaterialIds.data(), perDrawMaterialIds.size() * sizeof(uint32_t)))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload per-draw data");


	{
		std::vector<uint32_t> drawInfos;
		_uploadedBones = collectBones(*_scene, &drawInfos);
		if (drawInfos.empty())
			drawInfos.push_back(0);

		_resources.boneBuffer = renderApi::createStorageBuffer(_resources.gpu, _uploadedBones, renderApi::BufferUsage::DYNAMIC);
		_resources.drawInfoBuffer = renderApi::createStorageBuffer(_resources.gpu, drawInfos, renderApi::BufferUsage::DYNAMIC);
		if (!_resources.boneBuffer.isValid() || !_resources.drawInfoBuffer.isValid())
			return Result::error("SceneDescriptor '" + _name + "': failed to create bone buffers");
	}

	loadTextures();
	// Texture 0 is what materials without any albedo sample: the first file texture, else white.
	Result fallbackResult = ensureFallbackTexture();
	if (fallbackResult.code == ResultCode::Error)
		return fallbackResult;

	// The shaders only sample textures: a flat color is a 1x1 texture, shared by the materials using it.
	std::unordered_map<uint32_t, uint32_t> colorTextures;
	std::vector<uint32_t> materialData;
	materialData.reserve(materialTextureNames.size());
	for (size_t i = 0; i < materialTextureNames.size(); ++i) {
		const std::string& texName = materialTextureNames[i];
		uint32_t textureIndex = 0;
		auto it = _textureIndexMap.find(texName);
		if (!texName.empty() && it != _textureIndexMap.end()) {
			textureIndex = it->second;
		} else if (materialColors[i]) {
			const auto channel = [](float value) {
				return static_cast<uint32_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
			};
			const vec3& color = *materialColors[i];
			// Alpha is the opposite of the emission: stylized graphs read it (RenderGraphConfig::toon()),
			// the default one ignores it.
			const uint32_t pixel = channel(color.x) | (channel(color.y) << 8) | (channel(color.z) << 16) |
								   (channel(1.0f - materialEmissions[i]) << 24);

			auto colorIt = colorTextures.find(pixel);
			if (colorIt != colorTextures.end()) {
				textureIndex = colorIt->second;
			} else if (_resources.textures.size() < kMaxSceneTextures) {
				renderApi::Texture colorTex = renderApi::createTexture2D(
					_resources.gpu, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, &pixel, sizeof(pixel), false);
				if (colorTex.isValid()) {
					textureIndex = static_cast<uint32_t>(_resources.textures.size());
					_resources.textures.push_back(std::move(colorTex));
					colorTextures[pixel] = textureIndex;
				}
			}
		}
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
		matrices.push_back(drawCallMatrix(drawCall));
	});

	if (matrices.empty())
		return;

	// Nothing moved: the buffers and everything rendered from them are still valid.
	bool changed = false;
	if (matrices.size() != _uploadedMatrices.size() ||
		std::memcmp(matrices.data(), _uploadedMatrices.data(), matrices.size() * sizeof(mat4)) != 0) {
		_resources.modelMatrixBuffer.upload(matrices.data(), matrices.size() * sizeof(mat4));
		_uploadedMatrices = std::move(matrices);
		changed = true;
	}

	// Poses: the set of skinned draws only changes with a reload, so the size always matches.
	if (_resources.boneBuffer.isValid()) {
		std::vector<mat4> bones = collectBones(*_scene);
		if (bones.size() == _uploadedBones.size() &&
			std::memcmp(bones.data(), _uploadedBones.data(), bones.size() * sizeof(mat4)) != 0) {
			_resources.boneBuffer.upload(bones.data(), bones.size() * sizeof(mat4));
			_uploadedBones = std::move(bones);
			changed = true;
		}
	}

	// Visibility: a hidden asset is taken out of the draw list (instance count 0), so its
	// vertices are not even processed. Its null matrix only covers the frame in between.
	if (_resources.drawCommandsBuffer().isValid() && !_drawCommands.empty()) {
		std::vector<bool> visible;
		visible.reserve(_drawVisible.size());
		_scene->forEachSubMesh([&](const DrawCallView& drawCall) { visible.push_back(drawCall.visible); });
		if (visible.size() == _drawVisible.size() && visible != _drawVisible) {
			for (size_t i = 0; i < visible.size(); ++i)
				_drawCommands[i].instanceCount = visible[i] ? 1 : 0;
			// Into another buffer than the one the frames in flight read (see SharedGpuResources).
			_resources.setDrawCommands(_drawCommands.data(), _drawCommands.size() * sizeof(VkDrawIndexedIndirectCommand));
			_drawVisible = std::move(visible);
			changed		 = true;
		}
	}

	// Materials: an asset may have switched to another one of the scene.
	if (_resources.perDrawBuffer.isValid() && !_uploadedMaterialIds.empty()) {
		std::vector<uint32_t> ids;
		ids.reserve(_uploadedMaterialIds.size());
		_scene->forEachSubMesh([&](const DrawCallView& drawCall) {
			const auto it = _materialIds.find(drawCall.material.name);
			ids.push_back(it != _materialIds.end() ? it->second : 0);
		});
		if (ids.size() == _uploadedMaterialIds.size() && ids != _uploadedMaterialIds) {
			_resources.perDrawBuffer.upload(ids.data(), ids.size() * sizeof(uint32_t));
			_uploadedMaterialIds = std::move(ids);
		}
	}

	if (changed)
		++_resources.geometryVersion;
}

bool SceneDescriptor::needsReload() const {
	return _isLoaded && _scene && _scene->getStructureVersion() != _loadedStructureVersion;
}

void SceneDescriptor::collectLights(std::vector<RasterCore::Light>& out) const {
	if (!_scene)
		return;

	// Suns first: they get the shadow map before plain directional lights.
	for (const auto& [name, asset] : _scene->getAssets()) {
		const auto* sun = asset.getType() == AssetType::Sun ? asset.getSunData() : nullptr;
		if (!sun || !asset.isVisible())
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
		if (!data || !asset.isVisible())
			continue;

		RasterCore::Light light;
		light.color = {data->color.x, data->color.y, data->color.z};
		light.intensity = data->intensity;
		light.castsShadow = true; // the renderer decides which ones actually get a shadow map

		// projection: 0 = Point, 1 = Directional, 2 = Spot (Asset::LightData is private)
		if (const auto* point = std::get_if<0>(&data->projection)) {
			light.type = RasterCore::LightType::Point;
			light.position = {point->position.x, point->position.y, point->position.z};
		} else if (const auto* directional = std::get_if<1>(&data->projection)) {
			light.type = RasterCore::LightType::Directional;
			light.direction = {directional->direction.x, directional->direction.y, directional->direction.z};
		} else if (const auto* spot = std::get_if<2>(&data->projection)) {
			light.type = RasterCore::LightType::Spot;
			light.position = {spot->position.x, spot->position.y, spot->position.z};
			light.direction = {spot->direction.x, spot->direction.y, spot->direction.z};
			light.coneAngle = spot->angle;
			light.coneBlend = spot->blend;
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

	// Nothing is skinned, but the shaders still read these.
	const std::vector<VertexSkin> skins(vertices.size());
	const std::vector<uint32_t>	  drawInfos = {0};
	_resources.skinBuffer = renderApi::createVertexBuffer<VertexSkin>(_resources.gpu, _resources.vertexCount);
	_resources.boneBuffer = renderApi::createStorageBuffer(_resources.gpu, matrices, renderApi::BufferUsage::DYNAMIC);
	_resources.drawInfoBuffer = renderApi::createStorageBuffer(_resources.gpu, drawInfos, renderApi::BufferUsage::DYNAMIC);
	if (!_resources.skinBuffer.isValid() || !_resources.boneBuffer.isValid() || !_resources.drawInfoBuffer.isValid() ||
		!_resources.skinBuffer.upload(skins.data(), skins.size() * sizeof(VertexSkin)))
		return Result::error("SceneDescriptor '" + _name + "': failed to create skinning buffers");

	VkDrawIndexedIndirectCommand cmd{};
	cmd.indexCount    = _resources.indexCount;
	cmd.instanceCount = 1;
	cmd.firstIndex    = 0;
	cmd.vertexOffset  = 0;
	cmd.firstInstance = 0;

	_resources.drawCount = 1;
	if (!_resources.createDrawCommands(&cmd, sizeof(cmd)))
		return Result::error("SceneDescriptor '" + _name + "': failed to create draw commands buffer");

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
		// An image file, on disk or kept in memory (textures embedded in a .glb).
		int w = 0, h = 0, channels = 0;
		stbi_uc* pixels = nullptr;
		std::string source;
		if (const auto* encoded = std::get_if<Texture::Encoded>(&tex.data)) {
			source = key;
			pixels = stbi_load_from_memory(encoded->bytes.data(), static_cast<int>(encoded->bytes.size()), &w, &h, &channels, STBI_rgb_alpha);
		} else if (const auto* file = std::get_if<Texture::FromFile>(&tex.data)) {
			if (file->path.empty())
				continue;
			source = file->path;
			pixels = stbi_load(file->path.c_str(), &w, &h, &channels, STBI_rgb_alpha);
		} else {
			continue;
		}
		if (!pixels) {
			std::cerr << "SceneDescriptor '" << _name << "': failed to load texture '"
					  << source << "' (" << stbi_failure_reason() << ")" << std::endl;
			continue;
		}

		// Larger images are halved until they fit: textures have no mip maps, so more texels
		// would only cost memory and shimmer.
		std::vector<stbi_uc> reduced;
		const stbi_uc* texels = pixels;
		while (w > kMaxTextureSize || h > kMaxTextureSize) {
			const int halfW = std::max(w / 2, 1), halfH = std::max(h / 2, 1);
			std::vector<stbi_uc> half(static_cast<size_t>(halfW) * halfH * 4);
			for (int y = 0; y < halfH; ++y) {
				for (int x = 0; x < halfW; ++x) {
					const int x1 = std::min(x * 2 + 1, w - 1), y1 = std::min(y * 2 + 1, h - 1);
					for (int c = 0; c < 4; ++c) {
						const int sum = texels[(static_cast<size_t>(y * 2) * w + x * 2) * 4 + c] + texels[(static_cast<size_t>(y * 2) * w + x1) * 4 + c] +
										texels[(static_cast<size_t>(y1) * w + x * 2) * 4 + c] + texels[(static_cast<size_t>(y1) * w + x1) * 4 + c];
						half[(static_cast<size_t>(y) * halfW + x) * 4 + c] = static_cast<stbi_uc>((sum + 2) / 4);
					}
				}
			}
			reduced = std::move(half);
			texels	= reduced.data();
			w = halfW;
			h = halfH;
		}

		renderApi::Texture gpuTex = renderApi::createTexture2D(
			_resources.gpu,
			static_cast<uint32_t>(w),
			static_cast<uint32_t>(h),
			VK_FORMAT_R8G8B8A8_UNORM,
			texels,
			static_cast<size_t>(w) * h * 4,
			false);

		stbi_image_free(pixels);

		if (!gpuTex.isValid()) {
			std::cerr << "SceneDescriptor '" << _name << "': failed to create GPU texture '"
					  << source << "'" << std::endl;
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
