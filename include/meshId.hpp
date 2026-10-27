#pragma once

#include <cstdint>

// A mesh loaded on the GPU (SceneManager::loadMesh()).
using MeshId = uint32_t;
inline constexpr MeshId kNoMesh = UINT32_MAX;
