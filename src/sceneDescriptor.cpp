#include <unordered_map>
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

namespace {

	// Copies into GPU buffers, gathered in one staging buffer and sent in one submission.
	class UploadBatch {
		private:
			struct Copy {
				VkBuffer	 target;
				VkDeviceSize offset;
				VkDeviceSize from;
				VkDeviceSize size;
			};
			std::vector<uint8_t> _bytes;
			std::vector<Copy>	 _copies;

		public:
			// `size` bytes of `data` (zeros when it is null) at `offset` of `target`.
			void add(const renderApi::Buffer& target, size_t offset, const void* data, size_t size) {
				if (size == 0)
					return;
				const size_t from = _bytes.size();
				_bytes.resize(from + size);
				if (data)
					std::memcpy(_bytes.data() + from, data, size);
				else
					std::memset(_bytes.data() + from, 0, size);
				_copies.push_back({target.getHandle(), offset, from, size});
			}

			bool submit(renderApi::device::GPU* gpu) {
				if (_copies.empty())
					return true;
				renderApi::Buffer staging = renderApi::createStagingBuffer(gpu, _bytes.size());
				void*			  to	  = staging.isValid() ? staging.map() : nullptr;
				if (!to)
					return false;
				std::memcpy(to, _bytes.data(), _bytes.size());
				staging.unmap();
				VkCommandBuffer cmd = gpu->beginOneTimeCommands();
				for (const Copy& copy : _copies) {
					const VkBufferCopy region{copy.from, copy.offset, copy.size};
					vkCmdCopyBuffer(cmd, staging.getHandle(), copy.target, 1, &region);
				}
				gpu->endOneTimeCommands(cmd); // waits for the copies
				return true;
			}
	};

	// A device buffer of `bytes`, holding what `buffer` held: made larger only when it is too small.
	bool growBuffer(renderApi::device::GPU* gpu, renderApi::Buffer& buffer, size_t bytes, renderApi::BufferType type) {
		if (buffer.isValid() && buffer.getSize() >= bytes)
			return true;
		renderApi::Buffer larger;
		if (!larger.create(gpu, std::max<size_t>(bytes, 16), type, renderApi::BufferUsage::STATIC))
			return false;
		if (buffer.isValid() && buffer.getSize() > 0) {
			VkCommandBuffer	   cmd = gpu->beginOneTimeCommands();
			const VkBufferCopy region{0, 0, buffer.getSize()};
			vkCmdCopyBuffer(cmd, buffer.getHandle(), larger.getHandle(), 1, &region);
			gpu->endOneTimeCommands(cmd);
		}
		buffer = std::move(larger);
		return true;
	}

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

	// Where `count` free elements are, the allocator grown (at least doubled) when it has no such room.
	uint32_t take(RangeAllocator& ranges, uint32_t count) {
		if (const auto first = ranges.allocate(count))
			return *first;
		ranges.grow(std::max(ranges.capacity() * 2, ranges.capacity() + count));
		return *ranges.allocate(count);
	}

}

Result SceneDescriptor::loadFromScene() {
	// Room for the scene and some more: assets added later go there, nothing is made again.
	const uint32_t vertices = _scene->getVertexAmount(), indices = _scene->getIndexAmount();
	const uint32_t vertexRoom = vertices + vertices / 4 + 4096, indexRoom = indices + indices / 4 + 12288;
	_geometry.clear();
	_vertexRanges.reset(vertexRoom);
	_indexRanges.reset(indexRoom);
	_textureIndexMap.clear();
	_sameImage.clear();
	_colorTextures.clear();
	_materialIds.clear();
	_materialSources.clear();
	_textureIdentities.clear();

	_resources.vertexBuffer = renderApi::createVertexBuffer<Vertex>(_resources.gpu, vertexRoom);
	_resources.skinBuffer	= renderApi::createVertexBuffer<VertexSkin>(_resources.gpu, vertexRoom);
	_resources.indexBuffer	= renderApi::createIndexBuffer<uint32_t>(_resources.gpu, indexRoom);
	if (!_resources.vertexBuffer.isValid() || !_resources.skinBuffer.isValid() || !_resources.indexBuffer.isValid())
		return Result::error("SceneDescriptor '" + _name + "': failed to create the geometry buffers");

	Result result = update();
	if (result.code == ResultCode::Error)
		return result;

	_isLoaded	 = true;
	_incremental = true;
	std::cout << "SceneDescriptor '" << _name << "': loaded ("
		  << _resources.vertexCount << " vertices, "
		  << _resources.indexCount << " indices, "
		  << _resources.drawCount << " draws, "
		  << _materialSources.size() << " materials, "
		  << _resources.textures.size() << " textures)" << std::endl;
	return Result::ok();
}

Result SceneDescriptor::update() {
	if (!_scene || !_resources.gpu)
		return Result::error("SceneDescriptor '" + _name + "': no scene or GPU to update from");

	// New geometry: whatever was rendered from the previous one (cached shadow maps) is stale.
	++_resources.geometryVersion;
	_loadedStructureVersion = _scene->getStructureVersion();

	Result result = placeGeometry();
	if (result.code != ResultCode::Error)
		result = updateMaterials();
	if (result.code != ResultCode::Error)
		result = buildDraws();
	if (result.code == ResultCode::Error)
		return result;
	_scene->loadHierarchy();
	return Result::ok();
}

bool SceneDescriptor::canUpdate() const {
	return _isLoaded && _incremental;
}

Result SceneDescriptor::placeGeometry() {
	using Key			= std::pair<const void*, uint64_t>;
	const auto keyOf	= [](const GeometryView& view) { return Key{view.shared.get(), view.mesh}; };

	// What the scene holds now, and what of it is not on the GPU yet.
	struct Added {
		Key key;
		std::shared_ptr<const ModelMesh> shared;
		const std::vector<Vertex>* vertices;
		const std::vector<VertexSkin>* skin;
		std::vector<const std::vector<uint32_t>*> indices;
	};
	std::map<Key, bool> present;
	std::vector<Added>	added;
	_scene->placeGeometry([&](const GeometryView& view) {
		const Key key = keyOf(view);
		if (present.emplace(key, true).second && _geometry.find(key) == _geometry.end())
			added.push_back({key, view.shared, &view.vertices, &view.skin, view.indices});
		return GeometryPlacement{};
	});

	// What left gives its room back first, so that what came may take it.
	size_t removed = 0;
	for (auto it = _geometry.begin(); it != _geometry.end();) {
		if (present.count(it->first)) {
			++it;
			continue;
		}
		_vertexRanges.release(it->second.firstVertex, it->second.vertexCount);
		for (const auto& [first, count] : it->second.indexRanges)
			_indexRanges.release(first, count);
		it = _geometry.erase(it);
		++removed;
	}

	// Room for what came, the allocators growing when they must; the buffers follow below.
	struct Upload {
		uint32_t firstVertex;
		const std::vector<Vertex>* vertices;
		const std::vector<VertexSkin>* skin;
		std::vector<std::pair<uint32_t, const std::vector<uint32_t>*>> indices;
	};
	std::vector<Upload> uploads;
	uploads.reserve(added.size());
	for (Added& piece : added) {
		GeometryRanges ranges;
		ranges.shared	   = std::move(piece.shared);
		ranges.vertexCount = static_cast<uint32_t>(piece.vertices->size());
		ranges.firstVertex = take(_vertexRanges, ranges.vertexCount);
		Upload upload{ranges.firstVertex, piece.vertices, piece.skin, {}};
		for (const std::vector<uint32_t>* list : piece.indices) {
			uint32_t first = 0;
			const auto same = std::find_if(upload.indices.begin(), upload.indices.end(), [&](const auto& placed) { return placed.second == list; });
			if (same != upload.indices.end()) {
				first = same->first;
			} else {
				const auto count = static_cast<uint32_t>(list->size());
				first			 = take(_indexRanges, count);
				ranges.indexRanges.push_back({first, count});
				upload.indices.push_back({first, list});
			}
			ranges.firstIndex.push_back(first);
		}
		uploads.push_back(std::move(upload));
		_geometry.emplace(piece.key, std::move(ranges));
	}

	if (!growBuffer(_resources.gpu, _resources.vertexBuffer, size_t(_vertexRanges.capacity()) * sizeof(Vertex), renderApi::BufferType::VERTEX) ||
		!growBuffer(_resources.gpu, _resources.skinBuffer, size_t(_vertexRanges.capacity()) * sizeof(VertexSkin), renderApi::BufferType::VERTEX) ||
		!growBuffer(_resources.gpu, _resources.indexBuffer, size_t(_indexRanges.capacity()) * sizeof(uint32_t), renderApi::BufferType::INDEX))
		return Result::error("SceneDescriptor '" + _name + "': failed to grow the geometry buffers");

	UploadBatch batch;
	for (const Upload& upload : uploads) {
		const size_t count = upload.vertices->size();
		batch.add(_resources.vertexBuffer, size_t(upload.firstVertex) * sizeof(Vertex), upload.vertices->data(), count * sizeof(Vertex));
		// Zeros for what is not skinned.
		batch.add(_resources.skinBuffer, size_t(upload.firstVertex) * sizeof(VertexSkin), upload.skin->size() == count ? upload.skin->data() : nullptr,
				  count * sizeof(VertexSkin));
		for (const auto& [first, list] : upload.indices)
			batch.add(_resources.indexBuffer, size_t(first) * sizeof(uint32_t), list->data(), list->size() * sizeof(uint32_t));
	}
	if (!batch.submit(_resources.gpu))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload the new geometry");

	// Every piece is on the GPU now: the scene learns where.
	_scene->placeGeometry([&](const GeometryView& view) {
		const GeometryRanges& ranges = _geometry.at(keyOf(view));
		return GeometryPlacement{ranges.firstVertex, ranges.firstIndex};
	});

	_resources.vertexCount = 0;
	_resources.indexCount  = 0;
	for (const auto& [key, ranges] : _geometry) {
		_resources.vertexCount += ranges.vertexCount;
		for (const auto& [first, count] : ranges.indexRanges)
			_resources.indexCount += count;
	}
	if (_isLoaded && (!added.empty() || removed > 0))
		std::cout << "SceneDescriptor '" << _name << "': " << added.size() << " pieces of geometry added, " << removed << " removed ("
				  << _resources.vertexCount << " of " << _vertexRanges.capacity() << " vertices used)" << std::endl;
	return Result::ok();
}

Result SceneDescriptor::updateMaterials() {
	// The image textures the scene got since the last time; texture 0 is what materials without
	// any albedo sample: the first file texture, else white.
	loadTextures();
	Result fallbackResult = ensureFallbackTexture();
	if (fallbackResult.code == ResultCode::Error)
		return fallbackResult;

	// Ids are kept: materials are only added. Those no draw uses yet get one too, so that an asset
	// may switch to one of them later (Asset::setMaterialOverride()) without anything being rebuilt.
	const auto add = [&](const Material& material) {
		if (_materialIds.find(material.name) != _materialIds.end())
			return;
		_materialIds[material.name] = static_cast<uint32_t>(_materialSources.size());
		_materialSources.push_back({materialTextureName(material), materialColor(material), materialEmission(material)});
	};
	_scene->forEachSubMesh([&](const DrawCallView& drawCall) { add(drawCall.material); });
	for (const auto& [name, material] : _scene->getMaterials())
		add(material);

	// The shaders only sample textures: a flat color is a 1x1 texture, shared by the materials using it.
	std::vector<uint32_t> materialData;
	materialData.reserve(_materialSources.size());
	for (const MaterialSource& source : _materialSources) {
		uint32_t textureIndex = 0;
		const auto it		  = _textureIndexMap.find(source.texture);
		if (!source.texture.empty() && it != _textureIndexMap.end()) {
			textureIndex = it->second;
		} else if (source.color) {
			const auto channel = [](float value) {
				return static_cast<uint32_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
			};
			const vec3& color = *source.color;
			// Alpha is the opposite of the emission: stylized graphs read it (RenderGraphConfig::toon()),
			// the default one ignores it.
			const uint32_t pixel = channel(color.x) | (channel(color.y) << 8) | (channel(color.z) << 16) |
								   (channel(1.0f - source.emission) << 24);

			auto colorIt = _colorTextures.find(pixel);
			if (colorIt != _colorTextures.end()) {
				textureIndex = colorIt->second;
			} else if (_resources.textures.size() < kMaxSceneTextures) {
				renderApi::Texture colorTex = renderApi::createTexture2D(
					_resources.gpu, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, &pixel, sizeof(pixel), false);
				if (colorTex.isValid()) {
					textureIndex = static_cast<uint32_t>(_resources.textures.size());
					_resources.textures.push_back(std::move(colorTex));
					_textureIdentities.push_back("");
					_colorTextures[pixel] = textureIndex;
				}
			}
		}
		materialData.push_back(textureIndex);
	}
	if (materialData.empty())
		materialData.push_back(0);

	const size_t bytes = materialData.size() * sizeof(uint32_t);
	if (!ensureRoom(_resources.gpu, _resources.materialBuffer, bytes, renderApi::BufferType::STORAGE, renderApi::BufferUsage::STATIC) ||
		!_resources.materialBuffer.upload(materialData.data(), bytes))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload the materials");
	return Result::ok();
}

Result SceneDescriptor::buildDraws() {
	// Everything per draw is made again: it is small, and the draws come in the order of the
	// assets, which an asset added or removed changes.
	std::vector<VkDrawIndexedIndirectCommand>& drawCommands = _drawCommands;
	drawCommands.clear();
	_drawVisible.clear();
	std::vector<mat4>	  matrices;
	std::vector<uint32_t> perDrawMaterialIds;
	_scene->forEachSubMesh([&](const DrawCallView& drawCall) {
		matrices.push_back(drawCallMatrix(drawCall));
		const auto it = _materialIds.find(drawCall.material.name);
		perDrawMaterialIds.push_back(it != _materialIds.end() ? it->second : 0);

		VkDrawIndexedIndirectCommand cmd{};
		cmd.indexCount	  = drawCall.indecesCount;
		cmd.instanceCount = drawCall.visible ? 1 : 0; // a hidden draw costs nothing: see syncModelMatrices()
		cmd.firstIndex	  = drawCall.indicesIndex;
		cmd.vertexOffset  = static_cast<int32_t>(drawCall.verticesIndex);
		cmd.firstInstance = 0;
		drawCommands.push_back(cmd);
		_drawVisible.push_back(drawCall.visible);
	});
	_uploadedMaterialIds = perDrawMaterialIds;
	_uploadedMatrices	 = matrices;
	if (matrices.empty())
		matrices.push_back(mat4::identity());

	using renderApi::BufferType;
	using renderApi::BufferUsage;
	renderApi::device::GPU* gpu = _resources.gpu;
	if (!ensureRoom(gpu, _resources.modelMatrixBuffer, matrices.size() * sizeof(mat4), BufferType::STORAGE, BufferUsage::DYNAMIC) ||
		!_resources.modelMatrixBuffer.upload(matrices.data(), matrices.size() * sizeof(mat4)))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload the model matrices");

	if (!ensureRoom(gpu, _resources.perDrawBuffer, perDrawMaterialIds.size() * sizeof(uint32_t), BufferType::STORAGE, BufferUsage::DYNAMIC) ||
		(!perDrawMaterialIds.empty() && !_resources.perDrawBuffer.upload(perDrawMaterialIds.data(), perDrawMaterialIds.size() * sizeof(uint32_t))))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload the per-draw data");

	std::vector<uint32_t> drawInfos;
	_uploadedBones = collectBones(*_scene, &drawInfos);
	if (drawInfos.empty())
		drawInfos.push_back(0);
	if (!ensureRoom(gpu, _resources.boneBuffer, _uploadedBones.size() * sizeof(mat4), BufferType::STORAGE, BufferUsage::DYNAMIC) ||
		!_resources.boneBuffer.upload(_uploadedBones.data(), _uploadedBones.size() * sizeof(mat4)) ||
		!ensureRoom(gpu, _resources.drawInfoBuffer, drawInfos.size() * sizeof(uint32_t), BufferType::STORAGE, BufferUsage::DYNAMIC) ||
		!_resources.drawInfoBuffer.upload(drawInfos.data(), drawInfos.size() * sizeof(uint32_t)))
		return Result::error("SceneDescriptor '" + _name + "': failed to upload the bones");

	// The ring of draw lists keeps its room; it is only made again, larger, when the list outgrows it.
	const size_t commandBytes = drawCommands.size() * sizeof(VkDrawIndexedIndirectCommand);
	const bool	 fits		  = _resources.drawCommandsBuffer().isValid() && _resources.drawCommandsBuffer().getSize() >= commandBytes;
	if (!fits) {
		if (!_resources.createDrawCommands(drawCommands.data(), commandBytes, commandBytes + commandBytes / 2))
			return Result::error("SceneDescriptor '" + _name + "': failed to create the draw commands");
	} else if (commandBytes > 0 && !_resources.setDrawCommands(drawCommands.data(), commandBytes)) {
		return Result::error("SceneDescriptor '" + _name + "': failed to upload the draw commands");
	}
	_resources.drawCount = static_cast<uint32_t>(drawCommands.size());
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
	if (!_isLoaded)
		return Result::ok();

	if (_resources.gpu)
		vkDeviceWaitIdle(_resources.gpu->device);

	if (keepTextures)
		for (size_t i = 0; i < _resources.textures.size() && i < _textureIdentities.size(); ++i)
			if (!_textureIdentities[i].empty())
				_keptTextures[_textureIdentities[i]] = std::move(_resources.textures[i]);
	_textureIdentities.clear();
	_resources.destroy();
	_geometry.clear();
	_vertexRanges.reset(0);
	_indexRanges.reset(0);
	_textureIndexMap.clear();
	_sameImage.clear();
	_colorTextures.clear();
	_materialIds.clear();
	_materialSources.clear();
	_isLoaded	 = false;
	_incremental = false;
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
