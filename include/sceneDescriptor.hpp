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
