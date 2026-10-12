#pragma once

#include "Scene.hpp"
#include "rasterCore.hpp"
#include "return.hpp"

#include <string>
#include <unordered_map>
#include <vector>

class SceneDescriptor {
	private:
		std::string						_name;
		Scene*							_scene = nullptr;
		RasterCore::SharedGpuResources	_resources;
		bool							_isLoaded = false;
		std::unordered_map<std::string, uint32_t> _textureIndexMap;
		std::vector<mat4>				_uploadedMatrices; // model matrices currently on the GPU
		std::vector<mat4>				_uploadedBones;	   // bone matrices currently on the GPU
		std::unordered_map<std::string, uint32_t> _materialIds; // id of every material of the scene in the material buffer
		std::vector<uint32_t>			_uploadedMaterialIds;	 // material id of each draw currently on the GPU
		std::vector<VkDrawIndexedIndirectCommand> _drawCommands; // draw list currently on the GPU
		std::vector<bool>				_drawVisible;			 // which draws have an instance
		uint64_t						_loadedStructureVersion = 0; // Scene::getStructureVersion() the buffers were built from

		Result	loadFromScene();
		Result	loadFallbackPyramid();
		Result	loadTextures();
		Result	ensureFallbackTexture();

	public:
		SceneDescriptor() = default;
		SceneDescriptor(std::string name, Scene* scene, renderApi::device::GPU* gpu);
		~SceneDescriptor();

		SceneDescriptor(const SceneDescriptor&) = delete;
		SceneDescriptor& operator=(const SceneDescriptor&) = delete;
		SceneDescriptor(SceneDescriptor&&) = default;
		SceneDescriptor& operator=(SceneDescriptor&&) = default;

		Result								loadScene();
		Result								unloadScene();
		void								syncModelMatrices();
		// True when assets were added or removed since the GPU buffers were built.
		bool								needsReload() const;
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
