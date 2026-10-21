#pragma once

#include "Scene.hpp"
#include "rangeAllocator.hpp"
#include "rasterCore.hpp"
#include "return.hpp"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class SceneDescriptor {
	private:
		std::string						_name;
		Scene*							_scene = nullptr;
		RasterCore::SharedGpuResources	_resources;
		bool							_isLoaded = false;
		std::unordered_map<std::string, uint32_t> _textureIndexMap;
		std::vector<std::string>		_textureIdentities; // image of each texture of _resources (empty: a flat color or the fallback)
		std::unordered_map<std::string, renderApi::Texture> _keptTextures; // images of the last load, reused by the next one
		std::vector<mat4>				_uploadedMatrices; // model matrices currently on the GPU
		std::vector<mat4>				_uploadedBones;	   // bone matrices currently on the GPU
		std::unordered_map<std::string, uint32_t> _materialIds; // id of every material of the scene in the material buffer
		std::vector<uint32_t>			_uploadedMaterialIds;	 // material id of each draw currently on the GPU
		std::vector<VkDrawIndexedIndirectCommand> _drawCommands; // draw list currently on the GPU
		std::vector<bool>				_drawVisible;			 // which draws have an instance
		uint64_t						_loadedStructureVersion = 0; // Scene::getStructureVersion() the buffers were built from

		// Where each piece of geometry of the scene is in the vertex, skin and index buffers (see
		// update()), by who it is: (the shared geometry of a model, or none; the id of a mesh of
		// its own, or 0).
		struct GeometryRanges {
			std::shared_ptr<const ModelMesh> shared; // held while it is on the GPU: no other geometry takes its address
			uint32_t firstVertex = 0;
			uint32_t vertexCount = 0;
			std::vector<std::pair<uint32_t, uint32_t>> indexRanges; // first index and count of each list of it
			std::vector<uint32_t> firstIndex;						 // of each sub mesh
		};
		std::map<std::pair<const void*, uint64_t>, GeometryRanges> _geometry;
		RangeAllocator					_vertexRanges; // of the vertex and skin buffers
		RangeAllocator					_indexRanges;
		bool							_incremental = false; // built by loadFromScene(): update() keeps it current

		// The materials, by id (_materialIds): what their textures are made of and the numbers
		// they are multiplied by. See SceneDescriptor::updateMaterials().
		struct MaterialSource {
			std::string albedoTexture; // name of an image texture of the scene, or empty
			std::string normalTexture; // its blue channel is the normal map (tangent space)
			std::string ormTexture;	   // red = occlusion, green = roughness, blue = metallic
			vec3  tint{1.0f};		   // multiplied with the albedo (a texture, or the color)
			float roughness = 1.0f, metallic = 0.0f, occlusion = 1.0f;
			float normalIntensity = 1.0f;
			float uvScale		  = 1.0f;  // tiles per world unit (1 / Material::texture_scale)
			vec3  emission{0.0f};		   // color of what the surface gives off
			float emissionStrength = 0.0f; // 0..1, the "how lit" of the stylized graphs
			bool  cutout		   = false; // what the alpha of the albedo hides is not drawn at all
			float alphaCutoff	   = 0.5f;
		};
		std::vector<MaterialSource>				  _materialSources;
		std::unordered_map<std::string, uint32_t> _sameImage; // image identity -> its texture: the same image under several names

		Result	loadFromScene();
		Result	loadFallbackPyramid();
		Result	loadTextures();
		Result	ensureFallbackTexture();
		Result	placeGeometry();
		Result	updateMaterials();
		Result	buildDraws();

	public:
		SceneDescriptor() = default;
		SceneDescriptor(std::string name, Scene* scene, renderApi::device::GPU* gpu);
		~SceneDescriptor();

		SceneDescriptor(const SceneDescriptor&) = delete;
		SceneDescriptor& operator=(const SceneDescriptor&) = delete;
		SceneDescriptor(SceneDescriptor&&) = default;
		SceneDescriptor& operator=(SceneDescriptor&&) = default;

		Result								loadScene();
		// keepTextures: the image textures are kept for the next loadScene(), which takes back
		// those it still needs instead of decoding and uploading them again.
		Result								unloadScene(bool keepTextures = false);
		void								syncModelMatrices();
		// True when assets were added or removed since the GPU buffers were built.
		bool								needsReload() const;
		// Brings the GPU buffers to what the scene holds now, keeping what is already there: new
		// geometry is uploaded into free room, what left gives its room back, a buffer that is too
		// small grows (its content copied), textures and materials are only added. Nothing is
		// destroyed and made again. The caller holds GPU::GpuTasksMutex, so that no frame reads
		// the buffers meanwhile, then calls Viewport::refreshSceneBindings() on the viewports
		// showing the scene. After an error the buffers are not usable: unloadScene(), loadScene().
		Result								update();
		// Whether update() can follow the scene: loaded, and not the placeholder of an empty scene.
		bool								canUpdate() const;
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
