#pragma once

#include "Scene.hpp"
#include "meshId.hpp"
#include "rasterCore.hpp"
#include "return.hpp"

#include <string>
#include <unordered_map>
#include <vector>

class SceneManager;

class SceneDescriptor {
	private:
		std::string						_name;
		Scene*							_scene	 = nullptr;
		SceneManager*					_manager = nullptr;
		RasterCore::SharedGpuResources	_resources;
		bool							_isLoaded = false;
		uint64_t						_syncedVersion = ~0ull;

		struct Slot {
			MeshId	 mesh = kNoMesh;
			uint32_t list = 0;
		};

		struct GeometryRow {
			VkDeviceAddress vertices = 0, skin = 0, indices = 0, unused = 0;
		};
		std::vector<Slot>					   _slots;
		std::vector<uint32_t>				   _freeSlots;
		std::vector<mat4>					   _matrices;
		std::vector<uint32_t>				   _materials;
		std::vector<uint32_t>				   _drawInfos;
		std::vector<GeometryRow>			   _geometry;
		std::vector<VkDrawIndirectCommand>	   _commands;
		std::vector<mat4>					   _bones;
		bool								   _commitWanted = false;

		struct AssetSlots {
			std::vector<GeometryKey> geometry;
			std::vector<MeshId>		 meshes;
			std::vector<uint32_t>	 slots;
		};
		std::unordered_map<std::string, AssetSlots> _assets;
		std::vector<uint32_t>						_slotOrder;

		std::unordered_map<std::string, uint32_t> _textureIndexMap;
		std::vector<std::string>		_textureIdentities;
		std::unordered_map<std::string, renderApi::Texture> _keptTextures;
		std::unordered_map<std::string, uint32_t> _materialIds;

		struct MaterialSource {
			std::string albedoTexture;
			std::string normalTexture;
			std::string ormTexture;
			vec3  tint{1.0f};
			float roughness = 1.0f, metallic = 0.0f, occlusion = 1.0f;
			float normalIntensity = 1.0f;
			float uvScale		  = 1.0f;
			vec3  emission{0.0f};
			float emissionStrength = 0.0f;
			bool  cutout		   = false;
			float alphaCutoff	   = 0.5f;
		};
		std::vector<MaterialSource>				  _materialSources;
		std::unordered_map<std::string, uint32_t> _sameImage;

		Result	loadTextures();
		Result	ensureFallbackTexture();
		Result	updateMaterials();
		uint32_t takeSlot();
		// The rows of `slot` drawing list `list` of `mesh`: where its geometry is, and its command.
		void	writeGeometry(uint32_t slot);
		// Writes the rows that follow the scene each frame (matrix, material, bones, visibility)
		// of every slot; with `upload`, sends those that changed.
		void	readScene(bool upload);

	public:
		SceneDescriptor() = default;
		SceneDescriptor(std::string name, Scene* scene, renderApi::device::GPU* gpu, SceneManager* manager);
		~SceneDescriptor();

		SceneDescriptor(const SceneDescriptor&) = delete;
		SceneDescriptor& operator=(const SceneDescriptor&) = delete;
		SceneDescriptor(SceneDescriptor&&) = default;
		SceneDescriptor& operator=(SceneDescriptor&&) = default;

		// Everything the scene holds, from nothing (an empty scene loads too: no draw).
		Result								loadScene();
		// keepTextures: the image textures are kept for the next loadScene(), which takes back
		// those it still needs instead of decoding and uploading them again.
		Result								unloadScene(bool keepTextures = false);

		// An asset of the scene drawn: its meshes loaded (SceneManager::loadMesh()), a slot taken
		// for each of its sub meshes. removeAsset() gives them back. Both only change what is kept
		// here: commit() sends it.
		Result								addAsset(const std::string& name);
		void								removeAsset(const std::string& name);

		// True when assets were added or removed in the scene since the last sync().
		bool								needsReload() const;
		// Removes the assets that left the scene (or changed geometry) and adds those that came.
		// Nothing a frame reads is written. True when something changed: commit() must follow,
		// after SceneManager::flushUploads().
		Result								sync(bool& changed);
		// Sends the slots: new textures and materials, the per draw buffers (made larger when the
		// slots outgrew them) and the draw list. `geometryMoved`: the meshes may have moved
		// (compaction), the addresses of every slot are written again. The caller holds
		// GPU::GpuTasksMutex, so that no frame reads them meanwhile, then calls
		// Viewport::refreshSceneBindings() on the viewports showing the scene.
		Result								commit(bool geometryMoved);
		// Each frame: the matrices, materials, bones and visibility that changed.
		void								syncModelMatrices();
		// The rows outgrew a buffer since the last commit() (more bones): the next syncScenes()
		// commits, though no asset came or left.
		bool								commitWanted() const { return _commitWanted; }
		// Appends every sun and light asset of the scene, suns first (they get the cascaded shadows).
		void								collectLights(std::vector<RasterCore::Light>& out) const;

		void								setName(const std::string& name);
		const std::string&					getName() const;
		void								setScene(Scene* scene);
		Scene*								getScene() const;
		RasterCore::SharedGpuResources*		getResources();
		const RasterCore::SharedGpuResources* getResources() const;
		bool								isLoadedScene() const;
};
