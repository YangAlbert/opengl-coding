cmake_minimum_required(VERSION 3.0)
project(gl_geometry_shader)

set(platform_arch_path ${CMAKE_SYSTEM_NAME}/${CMAKE_GENERATOR_PLATFORM})

include_directories(external/include)
link_directories(external/lib/${platform_arch_path})

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
list(APPEND extra_libs glew32.lib glut32.lib)
message("windows extra libs: ${extra_libs}")
endif()

add_executable(gl_geometry_shader src/gl_geometry_shader.cpp)
target_link_libraries(gl_geometry_shader ${extra_libs})

message("platform ${CMAKE_SYSTEM_NAME}, arch: ${CMAKE_GENERATOR_PLATFORM}")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
message("cmake install dir: ${CMAKE_INSTALL_PREFIX}")
install(TARGETS gl_geometry_shader DESTINATION bin)
file(GLOB extra_dylibs external/bin/${platform_arch_path}/*)
message("extra dylibs: ${extra_dylibs}")
install(FILES ${extra_dylibs} DESTINATION bin)