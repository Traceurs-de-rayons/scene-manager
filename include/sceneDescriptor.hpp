#pragma once

#include "Scene.hpp"
#include "rasterCore.hpp"
#include "return.hpp"

#include <string>

class SceneDescriptor {
	private:
		std::string 					_name;
		Scene*							_scene;
		RasterCore::SharedGpuResources* _resources;
		bool							_isLoaded = false;

	public:
		SceneDescriptor() = default;
		SceneDescriptor(std::string name, Scene* scene, RasterCore::SharedGpuResources* resources);

		Result						loadScene();
		// bool						isLoadedScene();
		// Result						unloadScene();

		// Result

		void							setName(const std::string& name);
		const std::string&				getName() const;
		void							setScene(Scene* scene);
		Scene*							getScene() const;
		RasterCore::SharedGpuResources* getResources() const;
		bool							isLoadedScene() const;

};
