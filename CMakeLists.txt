cmake_minimum_required(VERSION 3.10)

project(mytest)

# debug
set(CMAKE_BUILD_TYPE "Debug")

set(name mytest)
set(SOURCE main.cpp)

add_executable(${name} ${SOURCE})

# 包含ncnn的头文件
include_directories("/root/WorkSpace/ncnn/build-android-aarch64/install/include/ncnn" "/root/WorkSpace/ncnn/src")
# 包含ncnn的链接文件
set(ncnn_DIR "/root/WorkSpace/ncnn/build-android-aarch64/install/lib/cmake/ncnn" CACHE PATH "/root/WorkSpace/ncnn/build-android-aarch64/install/lib/cmake/ncnn/ncnnConfig.cmake")

# x_86
# include_directories("/root/WorkSpace/ncnn/build/install/include/ncnn" "/root/WorkSpace/ncnn/src")
# set(ncnn_DIR "/root/WorkSpace/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "/root/WorkSpace/ncnn/build/install/lib/cmake/ncnn/ncnnConfig.cmake")

find_package(ncnn REQUIRED)   # 表示不找到 ncnn 就停掉 cmake
target_link_libraries(${name} ncnn)

set(OpenCV_DIR "/root/Softwares/opencv-mobile-4.8.1-android/sdk/native/jni/abi-arm64-v8a") # 包含OpenCVConfig.cmake
find_package(OpenCV REQUIRED)	# 找到opencv库
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${name} ${OpenCV_LIBRARIES})

# set(Vulkan_DIR "/root/Softwares/vulkansdk-linux-x86_64-1.2.189/source/VulkanTools/cmake/")
# # list(APPEND CMAKE_MODULE_PATH "/root/Softwares/vulkansdk-linux-x86_64-1.2.189/source/VulkanTools/cmake/")
# # set(CMAKE_PREFIX_PATH "/root/Softwares/vulkansdk-linux-x86_64-1.2.189/source/VulkanTools/cmake/" ${CMAKE_PREFIX_PATH})
# find_package(Vulkan REQUIRED)
# include_directories(${Vulkan_INCLUDE_DIRS})
# target_link_libraries(${name} Vulkan::Vulkan)

# cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
#     -DANDROID_ABI="arm64-v8a" \
#     -DANDROID_PLATFORM=android-24 ..