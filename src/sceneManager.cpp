#include "sceneManager.hpp"
#include "Scene.hpp"
#include "return.hpp"
#include "sceneDescriptor.hpp"
#include <cstdint>
#include <sys/types.h>

Result SceneManager::addScene(Scene* scene) {
	SceneDescriptor	descriptor;

	if (scene == nullptr)
		return Result::error("Scene is null");
	descriptor.setScene(scene);
	_scenesDesciptor.push_back(descriptor);
	return Result::ok();
}

void SceneManager::removeScene() { // trouver si pointeur / id / name
//
}

Result SceneManager::loadScene(const std::string& sceneName) {
	for (auto& descriptor : _scenesDesciptor) {
		if (descriptor.getName() == sceneName) {
			return (descriptor.loadScene());
		}
	}
	return Result::error("Scene not found");
}

Result SceneManager::loadScene(uint32_t	index) {
	if (index >= _scenesDesciptor.size())
		return Result::error("Index out of bounds");
	return (_scenesDesciptor[index].loadScene());
}

// Result SceneManager::unloadScene(const std::string& sceneName) {

// }
