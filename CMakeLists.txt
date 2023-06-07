cmake_minimum_required(VERSION 3.18)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 20)

project(main LANGUAGES CXX)

add_executable(main main.cpp)

find_package(OpenGL REQUIRED)
target_include_directories(main PUBLIC ${OPENGL_INCLUDE_DIR})
target_link_libraries(main PUBLIC ${OPENGL_LIBRARIES})

find_package(glm REQUIRED)
target_include_directories(main PUBLIC ${GLM_INCLUDE_DIRS})
target_link_directories(main PUBLIC ${GLM_LIBRARY_DIRS})

find_package(glfw3 REQUIRED)
target_include_directories(main PUBLIC ${GLFW_INCLUDE_DIRS})
target_link_libraries(main PUBLIC glfw)
