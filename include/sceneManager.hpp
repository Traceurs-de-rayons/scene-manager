#pragma once

#include "Scene.hpp"
#include "sceneDescriptor.hpp"
#include "return.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace renderApi::device {
	struct GPU;
}

class ViewportManager;

class SceneManager {
	private:
		std::vector<SceneDescriptor>	_scenesDescriptor;
		renderApi::device::GPU*			_gpu = nullptr;
		ViewportManager*				_viewportManager = nullptr;

		Result				reloadScene(SceneDescriptor& descriptor);

	public:
		SceneManager() = default;
		~SceneManager() = default;

		bool				init(renderApi::device::GPU* gpu);

		void				setViewportManager(ViewportManager* viewportManager) { _viewportManager = viewportManager; }

		Result				addScene(Scene* scene, const std::string& name);
		Result				removeScene(const std::string& name);

		Result				loadScene(const std::string& sceneName);
		Result				loadScene(uint32_t index);
		Result				unloadScene(const std::string& sceneName);

		// Rebuilds the GPU buffers of a loaded scene and rebinds the viewports showing it.
		Result				reloadScene(const std::string& sceneName);

		// Call once per frame: reloads the scenes whose assets were added or removed, then syncTransforms().
		void				syncScenes();
		void				syncTransforms();
		// All lights of all loaded scenes (each scene lists its suns first).
		std::vector<RasterCore::Light>	collectLights() const;

		SceneDescriptor*	getScene(const std::string& sceneName);
		SceneDescriptor*	getScene(uint32_t index);

		const std::vector<SceneDescriptor>&	getScenes() const { return _scenesDescriptor; }

		renderApi::device::GPU*				getGpu() const { return _gpu; }
};
