cmake_minimum_required(VERSION 3.0)
project(gl_geometry_shader)

include_directories(external/include)
link_directories(external/lib)

add_executable(gl_geometry_shader src/gl_geometry_shader.cpp)
target_link_libraries(gl_geometry_shader glew32.lib glut32.lib)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
message("cmake install dir: ${CMAKE_INSTALL_PREFIX}")
install(TARGETS gl_geometry_shader DESTINATION bin)
install(FILES external/bin/glut32.dll DESTINATION bin)
install(FILES external/bin/glew32.dll DESTINATION bin)