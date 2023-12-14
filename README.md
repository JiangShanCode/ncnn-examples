A c program of NCNN, which can SR the image and save the output(./res.png)
## How to build
```shell
cd ./build-android-aarch64

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
     -DANDROID_ABI="arm64-v8a" \
     -DANDROID_PLATFORM=android-24 ..

make -j8
```

## How to use
```
./mytest param=xxx.param bin=xxx.bin input=xxx.png gpu=1
```