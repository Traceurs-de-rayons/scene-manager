#include "sceneDescriptor.hpp"
#include "Scene.hpp"
#include "Vertex.hpp"
#include "renderApi.hpp"
#include "buffer.hpp"
#include "internal.hpp"
#include "return.hpp"
#include "viewportManager.hpp"
#include "vulkan/vulkan.hpp"
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <sys/types.h>

SceneDescriptor::SceneDescriptor(std::string name, Scene* scene, RasterCore::SharedGpuResources* resources)
		: _name(name), _scene(scene), _resources(resources) {}

Result SceneDescriptor::loadScene() {
	std::vector<mat4>	matrices;
	uint32_t			subMeshCount;


	_resources->indexCount = _scene->getIndexAmount();
	_resources->vertexCount = _scene->getVertexAmount();
	_resources->vertexBuffer = renderApi::createVertexBuffer<Vertex>(_resources->gpu, _resources->vertexCount);
	_resources->indexBuffer = renderApi::createIndexBuffer<uint32_t>(_resources->gpu, _resources->indexCount);

	matrices.clear();

	_scene->forEachSubMesh([&](const DrawCallView& drawCall)
	{
	    matrices.push_back(drawCall.transform.getMatrix());
	});

	subMeshCount = matrices.size();

	_resources->modelMatrixBuffer = renderApi::createUniformBuffer(
	    _resources->gpu,
	    subMeshCount * sizeof(mat4)
	);

	_resources->modelMatrixBuffer.update(matrices);
	return (Result::ok());
}



void						SceneDescriptor::setName(const std::string& name) { this->_name = name; }
const std::string&			SceneDescriptor::getName() const { return _name; }
void						SceneDescriptor::setScene(Scene* scene) { this->_scene = scene; }
Scene*						SceneDescriptor::getScene() const { return _scene; }
RasterCore::SharedGpuResources* SceneDescriptor::getResources() const { return _resources; }
bool						SceneDescriptor::isLoadedScene() const { return _isLoaded; }
