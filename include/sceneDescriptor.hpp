#pragma once

#include "Scene.hpp"
#include "rasterCore.hpp"

#include <string>

class SceneDescriptor {
	private:
		std::string 					name;
		Scene*							scene;
		RasterCore::SharedGpuResources	resources;
		bool							isLoaded = false;

	public:
		SceneDescriptor() = default;
		SceneDescriptor(std::string name, Scene* scene, RasterCore::SharedGpuResources resources)
		: name(name), scene(scene), resources(resources) {}

		bool						loadScene();
		bool						isLoaded();
		// Result
};
