#include "sceneManager.hpp"
#include "Scene.hpp"
#include "return.hpp"
#include "sceneDescriptor.hpp"
#include "viewport.hpp"
#include "viewportManager.hpp"

#include "renderDevice.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>

bool SceneManager::init(renderApi::device::GPU* gpu) {
	if (!gpu) {
		std::cerr << "SceneManager::init: GPU is null" << std::endl;
		return false;
	}
	_gpu = gpu;

	if (!_geometryPool.init(gpu)) {
		std::cerr << "SceneManager::init: failed to make the geometry buffer" << std::endl;
		return false;
	}
	return true;
}

Result SceneManager::addScene(Scene* scene, const std::string& name) {
	if (scene == nullptr)
		return Result::error("Scene is null");

	if (getScene(name) != nullptr)
		return Result::error("Scene '" + name + "' already exists");

	// The geometry of every scene in one library: what two scenes draw is kept once.
	if (scene->library() != _library)
		scene->setLibrary(_library);
	_scenesDescriptor.emplace_back(name, scene, _gpu, this);
	return Result::ok();
}

Result SceneManager::removeScene(const std::string& name) {
	for (auto it = _scenesDescriptor.begin(); it != _scenesDescriptor.end(); ++it) {
		if (it->getName() == name) {

			_scenesDescriptor.erase(it);
			return Result::ok();
		}
	}
	return Result::error("Scene '" + name + "' not found");
}

Result SceneManager::loadScene(const std::string& sceneName) {
	SceneDescriptor* descriptor = getScene(sceneName);
	if (!descriptor)
		return Result::error("Scene '" + sceneName + "' not found");
	return descriptor->loadScene();
}

Result SceneManager::loadScene(uint32_t index) {
	SceneDescriptor* descriptor = getScene(index);
	if (!descriptor)
		return Result::error("Index out of bounds");
	return descriptor->loadScene();
}

Result SceneManager::unloadScene(const std::string& sceneName) {
	SceneDescriptor* descriptor = getScene(sceneName);
	if (!descriptor)
		return Result::error("Scene '" + sceneName + "' not found");


	if (_viewportManager)
		_viewportManager->detachScene(descriptor->getResources());

	return descriptor->unloadScene();
}

SceneDescriptor* SceneManager::getScene(const std::string& sceneName) {
	for (auto& descriptor : _scenesDescriptor) {
		if (descriptor.getName() == sceneName)
			return &descriptor;
	}
	return nullptr;
}

SceneDescriptor* SceneManager::getScene(uint32_t index) {
	if (index >= _scenesDescriptor.size())
		return nullptr;
	return &_scenesDescriptor[index];
}

Result SceneManager::reloadScene(const std::string& sceneName) {
	SceneDescriptor* descriptor = getScene(sceneName);
	if (!descriptor)
		return Result::error("Scene '" + sceneName + "' not found");
	return reloadScene(*descriptor);
}

Result SceneManager::reloadScene(SceneDescriptor& descriptor) {
	RasterCore::SharedGpuResources* resources = descriptor.getResources();

	// The pipelines hold descriptor sets on the buffers about to be destroyed: drop them first.
	std::vector<Viewport*> attached;
	if (_viewportManager) {
		for (const auto& viewport : _viewportManager->getViewports()) {
			if (viewport && viewport->getSceneResources() == resources) {
				attached.push_back(viewport.get());
				viewport->setSceneResources(nullptr, "");
			}
		}
	}

	// The images rarely change with the geometry: they are kept rather than decoded again.
	Result result = descriptor.unloadScene(true);
	if (result.code != ResultCode::Error)
		result = descriptor.loadScene();
	if (result.code == ResultCode::Error)
		return result;

	for (Viewport* viewport : attached) {
		Result bound = viewport->setSceneResources(resources, descriptor.getName());
		if (bound.code == ResultCode::Error)
			result = bound;
	}
	return result;
}

bool SceneManager::placeMesh(UploadBatch& batch, const std::vector<Vertex>& vertices, const std::vector<VertexSkin>& skin,
							 const std::vector<const std::vector<uint32_t>*>& lists, LoadedMesh& mesh) {
	// One cell: the vertices, their skin (zeros for what is not skinned), then each index list.
	const size_t count = vertices.size();
	VkDeviceSize bytes = count * (sizeof(Vertex) + sizeof(VertexSkin));
	for (const std::vector<uint32_t>* list : lists)
		bytes += list->size() * sizeof(uint32_t);
	const std::optional<GeometryPool::Cell> cell = _geometryPool.allocate(bytes);
	if (!cell)
		return false;

	mesh.cell		 = *cell;
	mesh.vertexCount = static_cast<uint32_t>(count);
	mesh.listOffsets.clear();
	mesh.listCounts.clear();
	const VkBuffer target = cell->size > 0 ? _geometryPool.handle(cell->buffer) : VK_NULL_HANDLE;
	VkDeviceSize   at	  = cell->offset;
	if (target) {
		batch.add(target, at, vertices.data(), count * sizeof(Vertex));
		at += count * sizeof(Vertex);
		batch.add(target, at, skin.size() == count ? skin.data() : nullptr, count * sizeof(VertexSkin));
		at += count * sizeof(VertexSkin);
	}
	for (const std::vector<uint32_t>* list : lists) {
		mesh.listOffsets.push_back(at);
		mesh.listCounts.push_back(static_cast<uint32_t>(list->size()));
		if (target)
			batch.add(target, at, list->data(), list->size() * sizeof(uint32_t));
		at += list->size() * sizeof(uint32_t);
	}
	return true;
}

MeshId SceneManager::registerMesh(LoadedMesh mesh) {
	MeshId id = static_cast<MeshId>(_meshes.size());
	if (!_freeMeshes.empty()) {
		id = _freeMeshes.back();
		_freeMeshes.pop_back();
		_meshes[id] = std::move(mesh);
	} else {
		_meshes.push_back(std::move(mesh));
	}
	_meshIds.emplace(_meshes[id].key, id);
	return id;
}

Result SceneManager::loadModel(const std::string& name, Model model) {
	PendingLoad load;
	load.name	 = name;
	load.isModel = true;
	load.model	 = std::move(model);
	if (load.model.meshes.empty())
		return Result::error("SceneManager: the model '" + name + "' has no mesh");
	return uploadNamed(std::move(load));
}

Result SceneManager::loadMesh(const std::string& name, std::vector<Vertex> vertices, std::vector<uint32_t> indices) {
	bool mesh = !vertices.empty() && !indices.empty() && indices.size() % 3 == 0;
	for (size_t i = 0; mesh && i < indices.size(); ++i)
		mesh = indices[i] < vertices.size();
	if (!mesh)
		return Result::error("SceneManager: '" + name + "' is not a mesh (triangles into its vertices)");
	PendingLoad load;
	load.name	  = name;
	load.vertices = std::move(vertices);
	load.indices  = std::move(indices);
	return uploadNamed(std::move(load));
}

Result SceneManager::uploadNamed(PendingLoad load) {
	{
		const std::lock_guard<std::mutex> lock(_loadsMutex);
		if (load.name.empty() || !_names.insert(load.name).second)
			return Result::error("SceneManager: the name '" + load.name + "' is already loaded or loading");
	}
	const auto fail = [&](const std::string& message) {
		for (const LoadedMesh& mesh : load.meshes)
			_geometryPool.releaseNow(mesh.cell); // no draw ever read them
		const std::lock_guard<std::mutex> lock(_loadsMutex);
		_names.erase(load.name);
		return Result::error("SceneManager: " + message);
	};

	// The cells and their copies, on this thread: the pool is the only thing shared.
	UploadBatch							 batch;
	static const std::vector<VertexSkin> noSkin;
	if (load.isModel) {
		for (const ModelMesh& source : load.model.meshes) {
			LoadedMesh mesh;
			if (!source.vertices.empty() && !source.indices.empty() && !placeMesh(batch, source.vertices, source.skin, {&source.indices}, mesh))
				return fail("no GPU memory left for the model '" + load.name + "'");
			load.meshes.push_back(std::move(mesh));
		}
	} else {
		LoadedMesh mesh;
		if (!placeMesh(batch, load.vertices, noSkin, {&load.indices}, mesh))
			return fail("no GPU memory left for the mesh '" + load.name + "'");
		load.meshes.push_back(std::move(mesh));
	}
	if (!batch.submit(_gpu))
		return fail("failed to copy '" + load.name + "' to the GPU");

	const std::lock_guard<std::mutex> lock(_loadsMutex);
	_ready.push_back(std::move(load));
	return Result::ok();
}

void SceneManager::finishLoads() {
	std::vector<PendingLoad> ready;
	{
		const std::lock_guard<std::mutex> lock(_loadsMutex);
		ready.swap(_ready);
	}
	for (PendingLoad& load : ready) {
		const bool named = load.isModel ? _library->loadModel(load.name, std::move(load.model))
										: _library->loadMesh(load.name, std::move(load.vertices), std::move(load.indices));
		const std::vector<std::shared_ptr<const ModelMesh>> geometry =
			named ? _library->geometry(load.name) : std::vector<std::shared_ptr<const ModelMesh>>{};
		if (geometry.size() != load.meshes.size()) {
			std::cerr << "SceneManager: '" << load.name << "' could not be named in the library" << std::endl;
			for (const LoadedMesh& mesh : load.meshes)
				_geometryPool.releaseNow(mesh.cell);
			_library->unload(load.name);
			const std::lock_guard<std::mutex> lock(_loadsMutex);
			_names.erase(load.name);
			continue;
		}
		std::vector<MeshId>& ids = _named[load.name];
		for (size_t i = 0; i < geometry.size(); ++i) {
			LoadedMesh& mesh = load.meshes[i];
			mesh.key		 = GeometryKey{geometry[i].get(), 0};
			mesh.owner		 = geometry[i];
			mesh.users		 = 1;
			ids.push_back(registerMesh(std::move(mesh)));
		}
	}
}

std::vector<std::string> SceneManager::loadedNames() const {
	std::vector<std::string> names;
	for (const auto& [name, ids] : _named)
		names.push_back(name);
	std::sort(names.begin(), names.end());
	return names;
}

std::vector<SceneManager::MeshInfo> SceneManager::meshInfos() const {
	std::unordered_map<MeshId, std::string> loadedAs;
	for (const auto& [name, ids] : _named)
		for (const MeshId id : ids)
			loadedAs[id] = name;

	std::vector<MeshInfo> infos;
	for (MeshId id = 0; id < _meshes.size(); ++id) {
		const LoadedMesh& mesh = _meshes[id];
		if (mesh.users == 0)
			continue;
		MeshInfo info;
		info.id		  = id;
		info.name	  = mesh.owner && !mesh.owner->name.empty() ? mesh.owner->name : "mesh " + std::to_string(mesh.key.mesh);
		info.loadedAs = loadedAs.count(id) ? loadedAs[id] : "";
		info.users	  = mesh.users;
		info.vertices = mesh.vertexCount;
		for (const uint32_t count : mesh.listCounts)
			info.indices += count;
		info.cell  = mesh.cell;
		info.bytes = VkDeviceSize(info.vertices) * (sizeof(Vertex) + sizeof(VertexSkin)) + VkDeviceSize(info.indices) * sizeof(uint32_t);
		infos.push_back(std::move(info));
	}
	std::sort(infos.begin(), infos.end(), [](const MeshInfo& a, const MeshInfo& b) {
		return std::pair(a.cell.buffer, a.cell.offset) < std::pair(b.cell.buffer, b.cell.offset);
	});
	return infos;
}

bool SceneManager::findGeometry(const std::string& nameOrId, std::shared_ptr<const ModelMesh>& geometry) const {
	geometry.reset();
	if (!nameOrId.empty() && nameOrId[0] == '#') {
		char*			   end = nullptr;
		const unsigned long id = std::strtoul(nameOrId.c_str() + 1, &end, 10);
		if (end == nameOrId.c_str() + 1 || *end != '\0' || id >= _meshes.size() || _meshes[id].users == 0)
			return false;
		geometry = _meshes[id].owner;
		return true;
	}
	for (const LoadedMesh& mesh : _meshes)
		if (mesh.users > 0 && mesh.owner && mesh.owner->name == nameOrId) {
			geometry = mesh.owner;
			return true;
		}
	return false;
}

Result SceneManager::unload(const std::string& name) {
	{
		const std::lock_guard<std::mutex> lock(_loadsMutex);
		if (!_names.count(name))
			return Result::error("SceneManager: '" + name + "' is not loaded");
	}
	if (!_named.count(name))
		return Result::error("SceneManager: '" + name + "' is still loading, unload cancelled");
	if (const uint32_t users = _library->users(name); users > 0)
		return Result::error("SceneManager: '" + name + "' is still used by " + std::to_string(users) + " assets, unload cancelled");

	for (const MeshId id : _named[name])
		unloadMesh(id);
	_named.erase(name);
	_library->unload(name);
	{
		const std::lock_guard<std::mutex> lock(_loadsMutex);
		_names.erase(name);
	}
	// No asset draws it, and outside syncScenes() no draw list is about to change: no frame reads
	// its cells, they are free at once.
	_geometryPool.freeReleased();
	return Result::ok();
}

MeshId SceneManager::loadMesh(const MeshData& mesh) {
	if (const auto it = _meshIds.find(mesh.key); it != _meshIds.end()) {
		++_meshes[it->second].users;
		return it->second;
	}
	if (!mesh.vertices || !mesh.skin)
		return kNoMesh;
	LoadedMesh loaded{mesh.key, mesh.owner, {}, 0, {}, {}, 1};
	if (!placeMesh(_uploads, *mesh.vertices, *mesh.skin, mesh.indices, loaded))
		return kNoMesh;
	return registerMesh(std::move(loaded));
}

void SceneManager::unloadMesh(MeshId mesh) {
	if (mesh == kNoMesh || mesh >= _meshes.size() || _meshes[mesh].users == 0)
		return;
	LoadedMesh& loaded = _meshes[mesh];
	if (--loaded.users > 0)
		return;
	_geometryPool.release(loaded.cell);
	_meshIds.erase(loaded.key);
	loaded = LoadedMesh{};
	_freeMeshes.push_back(mesh);
}

Result SceneManager::flushUploads() {
	if (!_uploads.submit(_gpu))
		return Result::error("SceneManager: failed to copy the geometry to the GPU");
	return Result::ok();
}

VkDeviceAddress SceneManager::meshVertices(MeshId mesh) const {
	const LoadedMesh& loaded = _meshes[mesh];
	return loaded.cell.size > 0 ? _geometryPool.address(loaded.cell.buffer) + loaded.cell.offset : 0;
}

VkDeviceAddress SceneManager::meshSkin(MeshId mesh) const {
	const VkDeviceAddress vertices = meshVertices(mesh);
	return vertices ? vertices + VkDeviceSize(_meshes[mesh].vertexCount) * sizeof(Vertex) : 0;
}

VkDeviceAddress SceneManager::meshIndices(MeshId mesh, uint32_t list) const {
	const LoadedMesh& loaded = _meshes[mesh];
	return loaded.cell.size > 0 ? _geometryPool.address(loaded.cell.buffer) + loaded.listOffsets[list] : 0;
}

uint32_t SceneManager::meshIndexCount(MeshId mesh, uint32_t list) const {
	const LoadedMesh& loaded = _meshes[mesh];
	return loaded.cell.size > 0 ? loaded.listCounts[list] : 0;
}

bool SceneManager::compactMeshes() {
	// Never in the same submission as a mesh not copied yet: it would be moved from an empty cell.
	if (!_uploads.empty() || !_geometryPool.wantsCompaction())
		return false;
	_geometryPool.compactionStarted();

	// The meshes of the last buffers first: they take the lowest free cells.
	std::vector<MeshId> order;
	for (MeshId id = 0; id < _meshes.size(); ++id)
		if (_meshes[id].users > 0 && _meshes[id].cell.size > 0)
			order.push_back(id);
	std::sort(order.begin(), order.end(), [&](MeshId a, MeshId b) {
		return std::pair(_meshes[a].cell.buffer, _meshes[a].cell.offset) > std::pair(_meshes[b].cell.buffer, _meshes[b].cell.offset);
	});

	size_t moved = 0;
	for (const MeshId id : order) {
		LoadedMesh& mesh = _meshes[id];
		const std::optional<GeometryPool::Cell> to = _geometryPool.relocate(mesh.cell);
		if (!to)
			continue;
		VkDeviceSize bytes = VkDeviceSize(mesh.vertexCount) * (sizeof(Vertex) + sizeof(VertexSkin));
		for (const uint32_t count : mesh.listCounts)
			bytes += VkDeviceSize(count) * sizeof(uint32_t);
		_uploads.move(_geometryPool.handle(mesh.cell.buffer), mesh.cell.offset, _geometryPool.handle(to->buffer), to->offset, bytes);
		for (VkDeviceSize& offset : mesh.listOffsets)
			offset = offset - mesh.cell.offset + to->offset;
		_geometryPool.release(mesh.cell);
		mesh.cell = *to;
		++moved;
	}
	if (moved > 0)
		std::cout << "SceneManager: " << moved << " meshes moved" << std::endl;
	return moved > 0;
}

void SceneManager::syncScenes() {
	// What was loaded by name since the last frame can be used from now on.
	finishLoads();

	// The meshes move first (compaction): every scene then writes its addresses again.
	const bool moved = compactMeshes();

	// The assets that came and left: meshes loaded and unloaded, slots taken and given back.
	std::vector<SceneDescriptor*> changed, failed;
	for (auto& descriptor : _scenesDescriptor) {
		if (!descriptor.isLoadedScene())
			continue;
		bool		 added	= false;
		const Result result = descriptor.sync(added);
		if (result.code == ResultCode::Error) {
			std::cerr << "SceneManager: " << result.message << ", loading the scene again" << std::endl;
			failed.push_back(&descriptor);
		} else if (added || moved || descriptor.commitWanted()) {
			changed.push_back(&descriptor);
		}
	}

	// The new geometry goes into cells no frame reads: the frames go on meanwhile.
	if (const Result result = flushUploads(); result.code == ResultCode::Error) {
		std::cerr << result.message << ", loading the scenes again" << std::endl;
		failed.insert(failed.end(), changed.begin(), changed.end());
		changed.clear();
	}

	// The per draw buffers, while no frame of a viewport runs: the GPU thread holds this lock
	// while it records and runs a frame. Then the cells given back are free (nothing draws from
	// them any more), and the viewports take the buffers that were made larger.
	if (!changed.empty()) {
		const auto lock = _gpu->lockTasks();
		for (SceneDescriptor* descriptor : changed) {
			const Result result = descriptor->commit(moved);
			if (result.code == ResultCode::Error) {
				std::cerr << "SceneManager: " << result.message << ", loading the scene again" << std::endl;
				failed.push_back(descriptor);
				continue;
			}
			if (_viewportManager)
				for (const auto& viewport : _viewportManager->getViewports())
					if (viewport && viewport->getSceneResources() == descriptor->getResources())
						viewport->refreshSceneBindings();
		}
		_geometryPool.freeReleased();
	}

	for (SceneDescriptor* descriptor : failed) {
		const Result result = reloadScene(*descriptor);
		if (result.code == ResultCode::Error)
			std::cerr << "SceneManager: " << result.message << std::endl;
	}
	syncTransforms();
}

void SceneManager::syncTransforms() {
	for (auto& descriptor : _scenesDescriptor)
		descriptor.syncModelMatrices();
}

std::vector<RasterCore::Light> SceneManager::collectLights() const {
	std::vector<RasterCore::Light> lights;
	for (const auto& descriptor : _scenesDescriptor)
		descriptor.collectLights(lights);
	return lights;
}
