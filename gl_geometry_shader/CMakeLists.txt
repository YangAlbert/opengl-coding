cmake_minimum_required(VERSION 3.0)
project(gl_geometry_shader_glfw)

# find_package(GLFW REQUIRED)
# find_package(GLEW REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenGL REQUIRED)

PKG_SEARCH_MODULE(GLFW REQUIRED glfw3)
PKG_SEARCH_MODULE(GLEW REQUIRED glew)

INCLUDE_DIRECTORIES(${GLFW_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})
MESSAGE(STATUS "FIND GLFW_INCLUDE_DIRS " ${GLFW_INCLUDE_DIRS})

set(platform_arch_path ${CMAKE_SYSTEM_NAME}/${CMAKE_GENERATOR_PLATFORM})

include_directories(external/include)
# link_directories(external/lib/${platform_arch_path})
link_directories(${GLFW_LIBRARY_DIRS} ${GLEW_LIBRARY_DIRS})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gl_geometry_shader_glfw src/gl_geometry_shader_glfw.cpp src/Circle.cpp)
target_link_libraries(gl_geometry_shader_glfw ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})

message("platform ${CMAKE_SYSTEM_NAME}, arch: ${CMAKE_GENERATOR_PLATFORM}")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
message("cmake install dir: ${CMAKE_INSTALL_PREFIX}")
install(TARGETS gl_geometry_shader_glfw DESTINATION bin)
# file(GLOB extra_dylibs external/bin/${platform_arch_path}/*)
# message("extra dylibs: ${extra_dylibs}")
# install(FILES ${extra_dylibs} DESTINATION bin)