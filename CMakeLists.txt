cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo_linux)

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -s -O3")

if (CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH lib64)
else()
  set(LIB_ARCH lib)
endif()

# rga
set(RGA_DIR ${CMAKE_SOURCE_DIR}/3rdparty/rga)
include_directories(${RGA_DIR}/include)

# drm
set(DRM_DIR ${CMAKE_SOURCE_DIR}/3rdparty/drm)
include_directories(${DRM_DIR}/include)
include_directories(${DRM_DIR}/include/libdrm)

include_directories(${CMAKE_SOURCE_DIR}/include)

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/librknn_api)
include_directories(${RKNN_API_PATH}/include)
set(RKNN_API_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknn_api.so)

#stb
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/)

set(CMAKE_INSTALL_RPATH "lib")

# add_executable(yolov5_demo
#        src/yolo_detect.cc
#        src/drm_func.c
#        src/rga_func.c
#        src/postprocess.cc
#        src/main.cc)
# target_link_libraries(yolov5_demo ${RKNN_API_LIB} dl)

# 添加一个动态库目标
add_library(yolo_detect SHARED
        src/yolo_detect.cc
        src/drm_func.c
        src/rga_func.c
        src/postprocess.cc
        src/main.cc)

# 链接库
target_link_libraries(yolo_detect ${RKNN_API_LIB} dl)



