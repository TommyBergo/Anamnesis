cmake_minimum_required(VERSION 3.22.1)
project(usearch_jni)

# Prebuilt USearch C API library (arm64-v8a only - downloaded from
# https://github.com/unum-cloud/usearch/releases, usearch_android_arm64_2.26.2.zip), already
# packaged under app/src/main/jniLibs/arm64-v8a/ so Gradle bundles it into the APK; declared here
# too so this target can link against it at build time.
add_library(usearch_c SHARED IMPORTED)
set_target_properties(usearch_c PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libusearch_c.so
)

add_library(usearch_jni SHARED usearch_jni.cpp)

target_include_directories(usearch_jni PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(usearch_jni usearch_c log)
