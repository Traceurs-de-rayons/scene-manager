#include "sceneDescriptor.hpp"
#include "Scene.hpp"
#include "buffer.hpp"
#include "image.hpp"
#include "return.hpp"
#include "sceneManager.hpp"
#include "stb_image.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <optional>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan_core.h>

SceneDescriptor::SceneDescriptor(std::string name, Scene* scene, renderApi::device::GPU* gpu, SceneManager* manager)
		: _name(std::move(name)), _scene(scene), _manager(manager) {
	_resources.gpu = gpu;
}

SceneDescriptor::~SceneDescriptor() {
	unloadScene();
}

static constexpr size_t kMaxSceneTextures = 128;
static constexpr int kMaxTextureSize = 2048;

static std::optional<vec3> materialColor(const Material& material) {
	if (material.albedo && std::holds_alternative<cu::math::vec3>(material.albedo->value))
		return std::get<cu::math::vec3>(material.albedo->value);
	return std::nullopt;
}

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
static std::string textureNameOf(const std::optional<MaterialParam<float>>& param) {
	if (param && std::holds_alternative<std::string>(param->value))
		return std::get<std::string>(param->value);
	return "";
}

static float factorOf(const std::optional<MaterialParam<float>>& param, const std::optional<float>& factor, float fallback) {
	if (param && std::holds_alternative<float>(param->value))
		return std::get<float>(param->value);
	if (factor)
		return *factor;
	return fallback;
}

static mat4 transposeMatrix(const mat4& m) {
	mat4 r(0.0f);
	for (int row = 0; row < 4; ++row)
		for (int col = 0; col < 4; ++col)
			r.m[col][row] = m.m[row][col];
	return r;
}

static mat4 drawCallMatrix(const DrawCallView& drawCall) {
	if (!drawCall.visible)
		return mat4(0.0f);
	return transposeMatrix(drawCall.transform.getMatrix());
}

namespace {

	// Room for at least `bytes`: made again, half as large again, only when it is too small (what
	// it held is lost: the caller writes it all).
	bool ensureRoom(renderApi::device::GPU* gpu, renderApi::Buffer& buffer, size_t bytes, renderApi::BufferType type, renderApi::BufferUsage usage) {
		bytes = std::max<size_t>(bytes, 16);
		if (buffer.isValid() && buffer.getSize() >= bytes)
			return true;
		renderApi::Buffer larger;
		if (!larger.create(gpu, bytes + bytes / 2, type, usage))
			return false;
		buffer = std::move(larger);
		return true;
	}

}

Result SceneDescriptor::loadScene() {
	++_resources.geometryVersion;

	if (_isLoaded)
		return Result::ok();

	if (!_resources.gpu)
		return Result::error("SceneDescriptor '" + _name + "': no GPU set");

	if (!_scene)
		return Result::error("SceneDescriptor '" + _name + "': no scene set");

	if (!_manager)
		return Result::error("SceneDescriptor '" + _name + "': no scene manager set");

	// From nothing: every asset of the scene is added (an empty scene loads too: no draw).
	_textureIndexMap.clear();
	_sameImage.clear();
	_materialIds.clear();
	_materialSources.clear();
	_textureIdentities.clear();
	_syncedVersion = ~0ull;

	bool   changed = false;
	Result result  = sync(changed);
	if (result.code != ResultCode::Error)
		result = _manager->flushUploads();
	if (result.code != ResultCode::Error)
		result = commit(false);
	if (result.code == ResultCode::Error)
		return result;
	_manager->freeReleased();

	_isLoaded = true;
	std::cout << "SceneDescriptor '" << _name << "': loaded ("
		  << _assets.size() << " assets, "
		  << _manager->loadedMeshes() << " meshes on the GPU, "
		  << _resources.drawCount << " draws, "
		  << _materialSources.size() << " materials, "
		  << _resources.textures.size() << " textures)" << std::endl;
	return Result::ok();
}

uint32_t SceneDescriptor::takeSlot() {
	if (!_freeSlots.empty()) {
		const uint32_t slot = _freeSlots.back();
		_freeSlots.pop_back();
		return slot;
	}
	const auto slot = static_cast<uint32_t>(_slots.size());
	_slots.emplace_back();
	_matrices.push_back(mat4(0.0f));
	_materials.push_back(0);
	_drawInfos.push_back(0);
	_geometry.emplace_back();
	_commands.push_back(VkDrawIndirectCommand{0, 0, 0, slot});
	return slot;
}

void SceneDescriptor::writeGeometry(uint32_t slot) {
	const Slot& drawn = _slots[slot];
	if (drawn.mesh == kNoMesh) {
		_geometry[slot]			   = {};
		_commands[slot].vertexCount = 0;
		return;
	}
	_geometry[slot] = {_manager->meshVertices(drawn.mesh), _manager->meshSkin(drawn.mesh), _manager->meshIndices(drawn.mesh, drawn.list), 0};
	// Not indexed: vertex n of the draw is the one its index n names (shaders/scene_vertex.slang).
	// Its first instance is its slot: the row the shaders read.
	_commands[slot].vertexCount	  = _manager->meshIndexCount(drawn.mesh, drawn.list);
	_commands[slot].firstVertex	  = 0;
	_commands[slot].firstInstance = slot;
}

Result SceneDescriptor::addAsset(const std::string& name) {
	if (!_scene || !_manager)
		return Result::error("SceneDescriptor '" + _name + "': no scene or scene manager");
	if (_assets.count(name))
		return Result::ok();

	// Its meshes, each with the index lists of its sub meshes (they may share one), and the draws.
	struct Draw {
		size_t	 mesh;
		uint32_t list;
	};
	AssetSlots			  asset;
	std::vector<MeshData> meshes;
	std::vector<Draw>	  draws;
	_scene->forEachSubMesh(name, [&](const DrawCallView& drawCall) {
		asset.geometry.push_back(drawCall.geometry);
		if (meshes.empty() || meshes.back().key != drawCall.geometry)
			meshes.push_back({drawCall.geometry, drawCall.owner ? *drawCall.owner : nullptr, drawCall.vertices, drawCall.skin, {}});
		std::vector<const std::vector<uint32_t>*>& lists = meshes.back().indices;
		auto list = std::find(lists.begin(), lists.end(), drawCall.indices);
		if (list == lists.end())
			list = lists.insert(lists.end(), drawCall.indices);
		draws.push_back({meshes.size() - 1, static_cast<uint32_t>(list - lists.begin())});
	});
	if (draws.empty())
		return Result::ok(); // not an object (a light, a camera): nothing to draw

	for (const MeshData& mesh : meshes) {
		const MeshId id = _manager->loadMesh(mesh);
		if (id == kNoMesh) {
			for (const MeshId loaded : asset.meshes)
				_manager->unloadMesh(loaded);
			return Result::error("SceneDescriptor '" + _name + "': no GPU memory left for the geometry of '" + name + "'");
		}
		asset.meshes.push_back(id);
	}
	for (const Draw& draw : draws) {
		const uint32_t slot = takeSlot();
		_slots[slot]		= {asset.meshes[draw.mesh], draw.list};
		writeGeometry(slot);
		asset.slots.push_back(slot);
	}
	_assets.emplace(name, std::move(asset));
	return Result::ok();
}

void SceneDescriptor::removeAsset(const std::string& name) {
	const auto it = _assets.find(name);
	if (it == _assets.end())
		return;
	for (const uint32_t slot : it->second.slots) {
		_slots[slot] = {};
		writeGeometry(slot);
		_commands[slot].instanceCount = 0;
		_matrices[slot]				  = mat4(0.0f);
		_freeSlots.push_back(slot);
	}
	for (const MeshId mesh : it->second.meshes)
		_manager->unloadMesh(mesh);
	_assets.erase(it);
}

bool SceneDescriptor::needsReload() const {
	return _isLoaded && _scene && _scene->getStructureVersion() != _syncedVersion;
}

Result SceneDescriptor::sync(bool& changed) {
	changed = false;
	if (!_scene || _scene->getStructureVersion() == _syncedVersion)
		return Result::ok();

	// What the scene holds now: the geometry of each asset, in the order of its sub meshes, and
	// the asset of each sub mesh of the scene (the order the rows are read in each frame).
	std::unordered_map<std::string, std::vector<GeometryKey>> now;
	std::vector<std::pair<const std::string*, uint32_t>>	   order;
	_scene->forEachSubMesh([&](const DrawCallView& drawCall) {
		std::vector<GeometryKey>& geometry = now[*drawCall.asset];
		order.push_back({drawCall.asset, static_cast<uint32_t>(geometry.size())});
		geometry.push_back(drawCall.geometry);
	});

	// What left, or is made of something else now, goes; what came is added.
	std::vector<std::string> gone;
	for (const auto& [name, asset] : _assets)
		if (const auto it = now.find(name); it == now.end() || it->second != asset.geometry)
			gone.push_back(name);
	for (const std::string& name : gone)
		removeAsset(name);
	size_t added = 0;
	for (const auto& [name, geometry] : now) {
		if (_assets.count(name))
			continue;
		if (const Result result = addAsset(name); result.code == ResultCode::Error)
			return result;
		++added;
	}
	if (_isLoaded)
		std::cout << "SceneDescriptor '" << _name << "': " << added << " assets added, " << gone.size() << " removed (" << _manager->loadedMeshes()
				  << " meshes on the GPU)" << std::endl;

	_slotOrder.clear();
	_slotOrder.reserve(order.size());
	for (const auto& [asset, index] : order)
		_slotOrder.push_back(_assets.at(*asset).slots[index]);
	_syncedVersion = _scene->getStructureVersion();
	changed		   = true;
	return Result::ok();
}

void SceneDescriptor::readScene(bool upload) {
	if (!_scene || _slotOrder.empty())
		return;

	// Bones: the poses of the skinned assets one after the other (the sub meshes of an asset share
	// its pose), each draw pointing at the first of its own.
	std::vector<mat4>									   bones;
	std::unordered_map<const std::vector<mat4>*, uint32_t> firstBones;
	bool matrices = false, materials = false, infos = false, commands = false;
	size_t draw = 0;
	_scene->forEachSubMesh([&](const DrawCallView& drawCall) {
		if (draw >= _slotOrder.size())
			return;
		const uint32_t slot = _slotOrder[draw++];

		const mat4 matrix = drawCallMatrix(drawCall);
		if (std::memcmp(&matrix, &_matrices[slot], sizeof(mat4)) != 0) {
			_matrices[slot] = matrix;
			matrices		= true;
		}

		const auto	   material = _materialIds.find(drawCall.material.name);
		const uint32_t id		= material != _materialIds.end() ? material->second : 0;
		if (id != _materials[slot]) {
			_materials[slot] = id;
			materials		 = true;
		}

		uint32_t first = 0;
		if (drawCall.bones) {
			const auto [it, added] = firstBones.try_emplace(drawCall.bones, static_cast<uint32_t>(bones.size()));
			if (added)
				for (const mat4& bone : *drawCall.bones)
					bones.push_back(transposeMatrix(bone));
			first = it->second;
		}
		const uint32_t info = first | (drawCall.viewModel ? RasterCore::SharedGpuResources::kViewModelBit : 0u);
		if (info != _drawInfos[slot]) {
			_drawInfos[slot] = info;
			infos			 = true;
		}

		// A hidden draw is taken out of the list (no instance): its vertices are not even read.
		const uint32_t instances = drawCall.visible && _commands[slot].vertexCount > 0 ? 1 : 0;
		if (instances != _commands[slot].instanceCount) {
			_commands[slot].instanceCount = instances;
			commands					  = true;
		}
	});
	if (bones.empty())
		bones.push_back(mat4::identity());
	const bool posed = bones.size() != _bones.size() || std::memcmp(bones.data(), _bones.data(), bones.size() * sizeof(mat4)) != 0;
	if (posed)
		_bones = std::move(bones);

	if (!upload)
		return; // commit() sends everything
	// In place: the rows are only read by the frames, and the draw list goes into the next buffer
	// of its ring. More bones than their buffer holds (an asset posed for the first time): neither
	// they nor the rows pointing into them go now, or the shaders would read past the end of the
	// buffer (the GPU hangs); commit() makes it larger, under the lock, and sends both.
	const bool bonesFit = _resources.boneBuffer.isValid() && _resources.boneBuffer.getSize() >= _bones.size() * sizeof(mat4);
	if (!bonesFit && (posed || infos))
		_commitWanted = true;
	if (matrices)
		_resources.modelMatrixBuffer.upload(_matrices.data(), _matrices.size() * sizeof(mat4));
	if (materials)
		_resources.perDrawBuffer.upload(_materials.data(), _materials.size() * sizeof(uint32_t));
	if (infos && bonesFit)
		_resources.drawInfoBuffer.upload(_drawInfos.data(), _drawInfos.size() * sizeof(uint32_t));
	if (posed && bonesFit)
		_resources.boneBuffer.upload(_bones.data(), _bones.size() * sizeof(mat4));
	if (commands)
		_resources.setDrawCommands(_commands.data(), _commands.size() * sizeof(VkDrawIndirectCommand));
	if (matrices || commands || (posed && bonesFit))
		++_resources.geometryVersion;
}

void SceneDescriptor::syncModelMatrices() {
	// Rows follow the order of the sub meshes, which only holds for the assets sync() saw.
	if (!_isLoaded || needsReload())
		return;
	readScene(true);
}

Result SceneDescriptor::commit(bool geometryMoved) {
	if (!_scene || !_resources.gpu)
		return Result::error("SceneDescriptor '" + _name + "': no scene or GPU to commit to");

	if (geometryMoved)
		for (uint32_t slot = 0; slot < _slots.size(); ++slot)
			writeGeometry(slot);

	Result result = updateMaterials();
	if (result.code == ResultCode::Error)
		return result;
	readScene(false);

	// Every per draw buffer, made larger when the slots outgrew it (then bound again by the
	// viewports: Viewport::refreshSceneBindings()).
	using renderApi::BufferType;
	using renderApi::BufferUsage;
	renderApi::device::GPU* gpu	  = _resources.gpu;
	const auto				write = [&](renderApi::Buffer& buffer, const void* data, size_t bytes) {
		 return ensureRoom(gpu, buffer, bytes, BufferType::STORAGE, BufferUsage::DYNAMIC) && (bytes == 0 || buffer.upload(data, bytes));
	};
	if (!write(_resources.modelMatrixBuffer, _matrices.data(), _matrices.size() * sizeof(mat4)) ||
		!write(_resources.perDrawBuffer, _materials.data(), _materials.size() * sizeof(uint32_t)) ||
		!write(_resources.drawInfoBuffer, _drawInfos.data(), _drawInfos.size() * sizeof(uint32_t)) ||
		!write(_resources.drawGeometryBuffer, _geometry.data(), _geometry.size() * sizeof(GeometryRow)) ||
		!write(_resources.boneBuffer, _bones.data(), _bones.size() * sizeof(mat4)))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload the per draw buffers");

	// The ring of draw lists keeps its room; it is only made again, larger, when the list outgrows it.
	const size_t commandBytes = _commands.size() * sizeof(VkDrawIndirectCommand);
	const bool	 fits		  = _resources.drawCommandsBuffer().isValid() && _resources.drawCommandsBuffer().getSize() >= commandBytes;
	if (!fits) {
		if (!_resources.createDrawCommands(_commands.data(), commandBytes, commandBytes + commandBytes / 2))
			return Result::error("SceneDescriptor '" + _name + "': failed to create the draw commands");
	} else if (commandBytes > 0 && !_resources.setDrawCommands(_commands.data(), commandBytes)) {
		return Result::error("SceneDescriptor '" + _name + "': failed to upload the draw commands");
	}
	_resources.drawCount  = static_cast<uint32_t>(_commands.size());
	_resources.indexCount = 0;
	for (const VkDrawIndirectCommand& command : _commands)
		_resources.indexCount += command.vertexCount;

	// New geometry: whatever was rendered from the previous one (cached shadow maps) is stale.
	++_resources.geometryVersion;
	_commitWanted = false;
	_scene->loadHierarchy();
	return Result::ok();
}

Result SceneDescriptor::updateMaterials() {
	// Texture 0 is white: what a material without an image samples (its tint is then its color),
	// and what an image that failed to load falls back to. It comes first so that it stays 0.
	Result fallbackResult = ensureFallbackTexture();
	if (fallbackResult.code == ResultCode::Error)
		return fallbackResult;
	// The image textures the scene got since the last time.
	loadTextures();

	// Everything a material says, in the terms of the shaders. What the scene gives as numbers
	// needs no image: the fallback texture (white, and a flat normal) leaves them as they are.
	const auto describe = [](const Material& material) {
		MaterialSource source;

		// Albedo: an image, or a color (which the white fallback leaves untouched).
		if (const auto color = materialColor(material))
			source.tint = *color;
		source.albedoTexture = materialTextureName(material);
		if (material.albedo_factor)
			source.tint = source.tint * *material.albedo_factor;

		source.normalTexture   = material.normal_map ? *material.normal_map : "";
		source.normalIntensity = material.normal_intensity ? *material.normal_intensity : 1.0f;

		// Metallic, roughness and occlusion share one image, the way glTF packs them (red =
		// occlusion, green = roughness, blue = metallic): the first of the three that names a
		// texture is the one that holds all of them.
		const std::string occlusion = textureNameOf(material.ambient_occlusion);
		const std::string roughness = textureNameOf(material.roughness);
		const std::string metallic  = textureNameOf(material.metallic);
		source.ormTexture			= !occlusion.empty() ? occlusion : (!roughness.empty() ? roughness : metallic);
		source.occlusion			= factorOf(material.ambient_occlusion, material.occlusion_factor, 1.0f);
		source.roughness			= factorOf(material.roughness, material.roughness_factor, 1.0f);
		source.metallic				= factorOf(material.metallic, material.metallic_factor, 0.0f);

		if (material.emission_color && std::holds_alternative<vec3>(material.emission_color->value))
			source.emission = std::get<vec3>(material.emission_color->value);
		source.emissionStrength = materialEmission(material);

		source.cutout = material.alpha_mode && *material.alpha_mode == "cutout";
		if (material.alpha_cutoff)
			source.alphaCutoff = *material.alpha_cutoff;

		// The map gives its geometry UVs in metres: a material saying how many metres one tile
		// covers turns that into the tiles per metre the shader multiplies the UVs by.
		if (material.texture_scale && *material.texture_scale > 0.0f)
			source.uvScale = 1.0f / *material.texture_scale;
		return source;
	};

	// Ids are kept: materials are only added. Those no draw uses yet get one too, so that an asset
	// may switch to one of them later (Asset::setMaterialOverride()) without anything being rebuilt.
	const auto add = [&](const Material& material) {
		if (_materialIds.find(material.name) != _materialIds.end())
			return;
		_materialIds[material.name] = static_cast<uint32_t>(_materialSources.size());
		_materialSources.push_back(describe(material));
	};
	_scene->forEachSubMesh([&](const DrawCallView& drawCall) { add(drawCall.material); });
	for (const auto& [name, material] : _scene->getMaterials())
		add(material);

	// The layout mirrors GpuMaterial in raster-core/shaders/geometry.frag.slang: four rows of
	// four 32-bit values, so both sides read the same bytes.
	struct GpuMaterial {
		float	 tint[4];	  // rgb = albedo tint, a = tiles per world unit
		float	 params[4];	  // roughness, metallic, occlusion, normal intensity
		float	 emission[4]; // rgb = color, a = strength
		uint32_t textures[4]; // albedo, normal, orm, flags
	};
	static_assert(sizeof(GpuMaterial) == 64, "the shaders read 64 bytes per material");

	const auto textureIndex = [&](const std::string& name) {
		if (name.empty())
			return 0u;
		const auto it = _textureIndexMap.find(name);
		return it != _textureIndexMap.end() ? it->second : 0u;
	};

	std::vector<GpuMaterial> materialData;
	materialData.reserve(_materialSources.size());
	for (const MaterialSource& source : _materialSources) {
		GpuMaterial material{};
		material.tint[0] = source.tint.x;
		material.tint[1] = source.tint.y;
		material.tint[2] = source.tint.z;
		material.tint[3] = source.uvScale;
		material.params[0] = source.roughness;
		material.params[1] = source.metallic;
		material.params[2] = source.occlusion;
		material.params[3] = source.normalIntensity;
		material.emission[0] = source.emission.x;
		material.emission[1] = source.emission.y;
		material.emission[2] = source.emission.z;
		material.emission[3] = source.emissionStrength;
		material.textures[0] = textureIndex(source.albedoTexture);
		material.textures[1] = textureIndex(source.normalTexture);
		material.textures[2] = textureIndex(source.ormTexture);
		material.textures[3] = (source.normalTexture.empty() ? 0u : 1u) | (source.ormTexture.empty() ? 0u : 2u) | (source.cutout ? 4u : 0u);
		materialData.push_back(material);
	}
	if (materialData.empty())
		materialData.push_back(GpuMaterial{});

	const size_t bytes = materialData.size() * sizeof(GpuMaterial);
	if (!ensureRoom(_resources.gpu, _resources.materialBuffer, bytes, renderApi::BufferType::STORAGE, renderApi::BufferUsage::STATIC) ||
		!_resources.materialBuffer.upload(materialData.data(), bytes))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload the materials");
	return Result::ok();
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

Result SceneDescriptor::loadTextures() {
	if (!_scene)
		return Result::ok();

	// One identity per texture of _resources, "" for what is not an image (a flat color, the fallback).
	_textureIdentities.resize(_resources.textures.size());

	// What the last load kept and this one does not take back is no longer used.
	struct DropUnused {
		std::unordered_map<std::string, renderApi::Texture>& kept;
		~DropUnused() {
			for (auto& [identity, texture] : kept)
				texture.destroy();
			kept.clear();
		}
	} dropUnused{_keptTextures};

	const auto& sceneTextures = _scene->getTextures();
	if (sceneTextures.empty())
		return Result::ok();

	// The same image under several names (a model added to another one copies its textures, the
	// same file taken by two models) is uploaded once: the shaders hold kMaxSceneTextures at most.
	// Only what is not on the GPU yet is looked at: textures are never removed from a scene.
	std::unordered_map<std::string, uint32_t>& sameImage = _sameImage;
	for (const auto& [key, tex] : sceneTextures) {
		if (_textureIndexMap.find(key) != _textureIndexMap.end())
			continue;
		std::string identity;
		if (const auto* encoded = std::get_if<Texture::Encoded>(&tex.data)) {
			uint64_t hash = 1469598103934665603ull;
			for (const uint8_t byte : encoded->bytes)
				hash = (hash ^ byte) * 1099511628211ull;
			identity = "data:" + std::to_string(encoded->bytes.size()) + ":" + std::to_string(hash);
		} else if (const auto* file = std::get_if<Texture::FromFile>(&tex.data)) {
			identity = "file:" + file->path;
		}
		if (const auto same = sameImage.find(identity); !identity.empty() && same != sameImage.end()) {
			_textureIndexMap[key] = same->second;
			continue;
		}
		if (_resources.textures.size() >= kMaxSceneTextures) {
			std::cerr << "SceneDescriptor '" << _name << "': more than " << kMaxSceneTextures << " textures, '" << key << "' left out" << std::endl;
			_textureIndexMap[key] = 0; // the fallback, and not tried again
			continue;
		}
		if (const auto kept = _keptTextures.find(identity); !identity.empty() && kept != _keptTextures.end()) {
			_textureIndexMap[key] = static_cast<uint32_t>(_resources.textures.size());
			sameImage[identity]	  = _textureIndexMap[key];
			_resources.textures.push_back(std::move(kept->second));
			_textureIdentities.push_back(identity);
			_keptTextures.erase(kept);
			continue;
		}

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
			_textureIndexMap[key] = 0;
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
			_textureIndexMap[key] = 0;
			continue;
		}

		_textureIndexMap[key] = static_cast<uint32_t>(_resources.textures.size());
		if (!identity.empty())
			sameImage[identity] = _textureIndexMap[key];
		_resources.textures.push_back(std::move(gpuTex));
		_textureIdentities.push_back(identity);
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
	_textureIdentities.push_back("");
	return Result::ok();
}

Result SceneDescriptor::unloadScene(bool keepTextures) {
	if (!keepTextures) {
		for (auto& [identity, texture] : _keptTextures)
			texture.destroy();
		_keptTextures.clear();
	}
	if (!_isLoaded && _assets.empty())
		return Result::ok();

	if (_resources.gpu)
		vkDeviceWaitIdle(_resources.gpu->device);

	// Nothing draws any more: the meshes are unloaded, their cells free at once.
	std::vector<std::string> names;
	for (const auto& [name, asset] : _assets)
		names.push_back(name);
	for (const std::string& name : names)
		removeAsset(name);
	if (_manager)
		_manager->freeReleased();

	if (keepTextures)
		for (size_t i = 0; i < _resources.textures.size() && i < _textureIdentities.size(); ++i)
			if (!_textureIdentities[i].empty())
				_keptTextures[_textureIdentities[i]] = std::move(_resources.textures[i]);
	_textureIdentities.clear();
	_resources.destroy();
	_slots.clear();
	_freeSlots.clear();
	_matrices.clear();
	_materials.clear();
	_drawInfos.clear();
	_geometry.clear();
	_commands.clear();
	_bones.clear();
	_slotOrder.clear();
	_textureIndexMap.clear();
	_sameImage.clear();
	_materialIds.clear();
	_materialSources.clear();
	_syncedVersion = ~0ull;
	_isLoaded	   = false;
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
