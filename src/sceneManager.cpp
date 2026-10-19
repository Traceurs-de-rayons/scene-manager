#include "sceneManager.hpp"
#include "Scene.hpp"
#include "return.hpp"
#include "sceneDescriptor.hpp"
#include "viewport.hpp"
#include "viewportManager.hpp"

#include "renderDevice.hpp"

#include <cstdint>
#include <iostream>
#include <mutex>

bool SceneManager::init(renderApi::device::GPU* gpu) {
	if (!gpu) {
		std::cerr << "SceneManager::init: GPU is null" << std::endl;
		return false;
	}
	_gpu = gpu;
	return true;
}

Result SceneManager::addScene(Scene* scene, const std::string& name) {
	if (scene == nullptr)
		return Result::error("Scene is null");

	if (getScene(name) != nullptr)
		return Result::error("Scene '" + name + "' already exists");

	_scenesDescriptor.emplace_back(name, scene, _gpu);
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

Result SceneManager::updateScene(SceneDescriptor& descriptor) {
	RasterCore::SharedGpuResources* resources = descriptor.getResources();
	if (!resources->gpu)
		return Result::error("Scene '" + descriptor.getName() + "' has no GPU");

	// The GPU thread holds this lock while it records and runs a frame of each viewport: from here
	// on nothing reads the scene buffers, which may be grown, written or replaced.
	const auto lock = resources->gpu->lockTasks();
	Result result = descriptor.update();
	if (result.code == ResultCode::Error)
		return result;
	if (_viewportManager)
		for (const auto& viewport : _viewportManager->getViewports())
			if (viewport && viewport->getSceneResources() == resources)
				viewport->refreshSceneBindings();
	return result;
}

void SceneManager::syncScenes() {
	for (auto& descriptor : _scenesDescriptor) {
		if (!descriptor.needsReload())
			continue;

		// In place when it can; from scratch otherwise (the placeholder of an empty scene, a
		// failed update), which also rebuilds the viewports.
		bool updated = false;
		if (descriptor.canUpdate()) {
			const Result result = updateScene(descriptor);
			updated				= result.code != ResultCode::Error;
			if (!updated)
				std::cerr << "SceneManager: " << result.message << ", loading the scene again" << std::endl;
		}
		if (!updated) {
			const Result result = reloadScene(descriptor);
			if (result.code == ResultCode::Error)
				std::cerr << "SceneManager: " << result.message << std::endl;
		}
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
