cmake_minimum_required(VERSION 3.16)
project(scene-manager VERSION 0.1.0 LANGUAGES CXX)

# Collect sources
file(GLOB_RECURSE SCENE_MANAGER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create library
add_library(scene-manager STATIC ${SCENE_MANAGER_SOURCES})

# Specify the include directories
target_include_directories(scene-manager
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Propagate C++ standard requirements
target_compile_features(scene-manager PUBLIC cxx_std_23)

# Define dependencies
if(NOT TARGET scene-core)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../scene-core scene-core-build)
endif()

# Add dependency on raster-core
if(NOT TARGET raster-core)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../raster-core raster-core-build)
endif()

target_link_libraries(scene-manager PUBLIC scene-core raster-core)
