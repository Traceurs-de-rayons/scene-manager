#pragma once

#include "Scene.hpp"
#include "geometryPool.hpp"
#include "meshId.hpp"
#include "sceneDescriptor.hpp"
#include "return.hpp"
#include "uploadBatch.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace renderApi::device {
	struct GPU;
}

class ViewportManager;

// What loadMesh() puts on the GPU: vertices, their skin (empty: not skinned) and one list of
// triangles per sub mesh (several sub meshes may give the same list), all in one cell.
struct MeshData {
	GeometryKey								   key;	  // who the geometry is: loaded once
	std::shared_ptr<const ModelMesh>		   owner; // held while it is loaded (shared geometry)
	const std::vector<Vertex>*				   vertices = nullptr;
	const std::vector<VertexSkin>*			   skin		= nullptr;
	std::vector<const std::vector<uint32_t>*> indices;
};

class SceneManager {
	private:
		// Declared before the scenes: a scene unloads its meshes when it is destroyed.
		GeometryPool _geometryPool;

		struct LoadedMesh {
			GeometryKey						 key;
			std::shared_ptr<const ModelMesh> owner;
			GeometryPool::Cell				 cell;
			uint32_t						 vertexCount = 0;
			std::vector<VkDeviceSize>		 listOffsets; // of each index list, in the buffer of the cell
			std::vector<uint32_t>			 listCounts;
			uint32_t						 users = 0;	  // loadMesh() calls not unloaded yet; 0: a free id
		};
		// The geometry by name, shared by the scenes (Scene::setLibrary(), given by addScene()), and
		// the meshes kept on the GPU for each name loaded through loadModel() / loadMesh().
		std::shared_ptr<GeometryLibrary>						 _library = std::make_shared<GeometryLibrary>();
		std::unordered_map<std::string, std::vector<MeshId>>	 _named;

		// A load by name, uploaded by the thread that asked for it, named on the main thread
		// (finishLoads()): the library and the meshes are only ever touched there.
		struct PendingLoad {
			std::string				name;
			bool					isModel = false;
			Model					model;				 // a model
			std::vector<Vertex>		vertices;			 // or a mesh
			std::vector<uint32_t>	indices;
			std::vector<LoadedMesh> meshes;				 // where each mesh went
		};
		std::mutex						_loadsMutex; // over the two below, which any thread reads
		std::unordered_set<std::string> _names;		 // loaded or loading: taken
		std::vector<PendingLoad>		_ready;		 // uploaded, waiting for finishLoads()

		std::vector<LoadedMesh>			_meshes; // by MeshId
		std::vector<MeshId>				_freeMeshes;
		std::map<GeometryKey, MeshId>	_meshIds;
		UploadBatch						_uploads; // what loadMesh() and compaction copy, sent by flushUploads()

		std::vector<SceneDescriptor>	_scenesDescriptor;
		renderApi::device::GPU*			_gpu = nullptr;
		ViewportManager*				_viewportManager = nullptr;

		Result				reloadScene(SceneDescriptor& descriptor);
		// A cell for a mesh and the copies filling it, into `batch`; false when the GPU is full.
		// Only the pool is touched: from any thread.
		bool				placeMesh(UploadBatch& batch, const std::vector<Vertex>& vertices, const std::vector<VertexSkin>& skin,
									  const std::vector<const std::vector<uint32_t>*>& lists, LoadedMesh& mesh);
		MeshId				registerMesh(LoadedMesh mesh);
		Result				uploadNamed(PendingLoad load);
		// The loads by name that were uploaded: named in the library, their meshes counted.
		void				finishLoads();
		// Moves the meshes to free cells placed before theirs (GeometryPool::wantsCompaction());
		// true when one moved. Copied by the next flushUploads().
		bool				compactMeshes();

	public:
		SceneManager() = default;
		~SceneManager() = default;

		bool				init(renderApi::device::GPU* gpu);

		void				setViewportManager(ViewportManager* viewportManager) { _viewportManager = viewportManager; }

		// What a user loads by name: a model or a mesh, copied to the GPU then kept there until
		// unload(). From any thread (a loading thread, after reading the file): the copy is done
		// and waited for on the calling thread, which neither the main thread nor the frames wait
		// for. The name can be used from the next syncScenes() on (isLoaded()): any scene of this
		// manager then adds instances of it, Scene::addModel(asset, "2_male"),
		// Scene::addMesh(asset, "hole", material). The manager must outlive the call.
		Result				loadModel(const std::string& name, Model model);
		Result				loadMesh(const std::string& name, std::vector<Vertex> vertices, std::vector<uint32_t> indices);
		// Main thread. Cancelled (an error) while an asset of a scene is made from the name
		// (GeometryLibrary::users()), or while it is still loading.
		Result				unload(const std::string& name);
		// Main thread: loaded and named (usable by the scenes).
		bool				isLoaded(const std::string& name) const { return _named.count(name) > 0; }
		// Main thread: every name loaded, in alphabetical order.
		std::vector<std::string> loadedNames() const;

		// What is on the GPU, for a debug view: each mesh, where it is and who uses it.
		struct MeshInfo {
			MeshId			   id = kNoMesh;
			std::string		   name;	 // of the geometry (a mesh of a model, the asset that first added it), or "mesh <id>" for a mesh of its own
			std::string		   loadedAs; // the name it was loaded under (loadModel(), loadMesh()), or empty
			uint32_t		   users	= 0;
			uint32_t		   vertices = 0;
			uint32_t		   indices	= 0;
			GeometryPool::Cell cell;	  // buffer, offset and size of the cell
			VkDeviceSize	   bytes = 0; // what its data takes of the cell
		};
		std::vector<MeshInfo> meshInfos() const;
		// A mesh on the GPU by what meshInfos() says of it: "#<id>", or its name. False when there is
		// none; `geometry` is then null, and also when it is a mesh of a single asset (not shared,
		// over 65536 vertices), which nothing else can draw.
		bool				  findGeometry(const std::string& nameOrId, std::shared_ptr<const ModelMesh>& geometry) const;
		GeometryPool::Stats	  geometryStats() const { return _geometryPool.stats(); }
		// The library every scene of this manager uses: give it to a scene before filling it
		// (Scene::setLibrary()), or addScene() does.
		const std::shared_ptr<GeometryLibrary>& library() const { return _library; }

		// The GPU memory of the geometry, by id (what the scenes use). loadMesh() puts a mesh in a cell of the pool, or only
		// counts one more user when the same geometry (same key) is already loaded; unloadMesh()
		// counts one less, and gives the cell back with the last one. The data is copied by
		// flushUploads(); a cell given back is free once no draw reads it any more
		// (freeReleased(), after the draw lists were made again). kNoMesh when the GPU is full.
		MeshId				loadMesh(const MeshData& mesh);
		void				unloadMesh(MeshId mesh);
		Result				flushUploads();
		void				freeReleased() { _geometryPool.freeReleased(); }

		// Where a loaded mesh is, for the draws: its vertices, their skin, an index list and its count.
		VkDeviceAddress		meshVertices(MeshId mesh) const;
		VkDeviceAddress		meshSkin(MeshId mesh) const;
		VkDeviceAddress		meshIndices(MeshId mesh, uint32_t list) const;
		uint32_t			meshIndexCount(MeshId mesh, uint32_t list) const;
		uint32_t			meshVertexCount(MeshId mesh) const { return _meshes[mesh].vertexCount; }
		size_t				loadedMeshes() const { return _meshIds.size(); }

		Result				addScene(Scene* scene, const std::string& name);
		Result				removeScene(const std::string& name);

		Result				loadScene(const std::string& sceneName);
		Result				loadScene(uint32_t index);
		Result				unloadScene(const std::string& sceneName);

		// Rebuilds the GPU buffers of a loaded scene and rebinds the viewports showing it.
		Result				reloadScene(const std::string& sceneName);

		// Call once per frame: the scenes whose assets were added or removed add and remove them
		// (SceneDescriptor::sync()), the new geometry is copied, then, while no frame of a
		// viewport runs, the per draw buffers are written (SceneDescriptor::commit()); a scene
		// that fails is loaded again. Then syncTransforms().
		void				syncScenes();
		void				syncTransforms();
		// All lights of all loaded scenes (each scene lists its suns first).
		std::vector<RasterCore::Light>	collectLights() const;

		SceneDescriptor*	getScene(const std::string& sceneName);
		SceneDescriptor*	getScene(uint32_t index);

		const std::vector<SceneDescriptor>&	getScenes() const { return _scenesDescriptor; }

		renderApi::device::GPU*				getGpu() const { return _gpu; }
};
