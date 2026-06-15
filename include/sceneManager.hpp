#pragma once

#include "Scene.hpp"
#include "sceneDescriptor.hpp"
#include <string>
#include <vector>

class SceneManager {
	private:
		std::vector<SceneDescriptor> _scenesDesciptor;

	public:
		SceneManager() = default;
		~SceneManager() = default;

		Result	addScene(Scene* descriptor);
		void	removeScene();
		Result	loadScene(const std::string& sceneName);
		Result	loadScene(uint32_t	index);

		bool	init();
};
