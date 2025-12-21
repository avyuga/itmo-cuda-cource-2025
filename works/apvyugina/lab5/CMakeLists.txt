cmake_minimum_required(VERSION 3.23)
project(TENSORRT_ONNX_DEMO)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)



# CUDA
if (NOT CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/lib/cuda/")
endif()
find_path(CUDA_INCLUDE_DIR cuda_runtime.h
  HINTS "/usr/include"
  PATH_SUFFIXES include
)
MESSAGE(STATUS "Found CUDA headers at ${CUDA_INCLUDE_DIR}")


# TensorRT
if (NOT TENSORRT_ROOT)
    set(TENSORRT_ROOT "/home/anastasia/TensorRT-10.14.1.48")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
MESSAGE(STATUS "Found TensorRT NVinfer library at ${TENSORRT_LIBRARY_INFER}")


find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
MESSAGE(STATUS "Found INFER_PLUGIN libs at ${TENSORRT_LIBRARY_INFER_PLUGIN}")

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
MESSAGE(STATUS "Found NvONNXParser at ${TENSORRT_LIBRARY_ONNXPARSER}")


# OpenCV
find_path(OPENCV_INCLUDE_DIRS NAMES "opencv2/core.hpp" HINTS "/usr/local/include/opencv4")
MESSAGE(STATUS "Found OpenCV headers at ${OPENCV_INCLUDE_DIRS}")

find_library(OPENCV_CORE_LIB NAMES "opencv_core" HINTS "/usr/local/lib")
MESSAGE(STATUS "Found OpenCV_CoreLib at ${OPENCV_CORE_LIB}")


set(CUDART "/usr/lib/x86_64-linux-gnu/libcudart.so")

set(PNG_LIB "/usr/lib/x86_64-linux-gnu/libpng16.so")
set(PNG_INCLUDE "/usr/include/")

add_compile_options(-std=c++17 -Wno-deprecated-declarations -Dcimg_display=0)
set_property(GLOBAL PROPERTY LINKER_LANGUAGE CXX)

### TRT Specific
set(TARGET_TRT "DetectionModelTRT")
add_executable(${TARGET_TRT} "${CMAKE_SOURCE_DIR}/src/DetectionModelTRT.cpp" "${CMAKE_SOURCE_DIR}/src/Utility.cpp")

target_link_libraries(${TARGET_TRT} 
  ${CUDART}
  ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_ONNXPARSER}
  ${OPENCV_CORE_LIB} ${PNG_LIB}
)

target_include_directories(${TARGET_TRT} 
    PUBLIC 
        ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR} ${OPENCV_INCLUDE_DIRS} ${PNG_INCLUDE}
    PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
)


