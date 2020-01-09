cmake_minimum_required(VERSION 2.8.10)
project(ceres-test C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(nlohmann_json 3.7.0 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_INSTALL_DIR}
)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

add_executable(${PROJECT_NAME} src/test-ceres.cpp)
target_link_libraries(${PROJECT_NAME}
    ${CERES_LIBRARIES}
)
