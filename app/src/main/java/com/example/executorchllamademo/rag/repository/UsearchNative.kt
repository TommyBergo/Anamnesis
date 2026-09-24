package com.example.executorchllamademo.rag.repository

/** Thin JNI bridge to USearch's flat C API, exposing only the native functions [UsearchVectorRepository] needs. */
object UsearchNative {
    init {
        System.loadLibrary("usearch_jni")
    }

    external fun nativeInit(dimensions: Int): Long

    external fun nativeReserve(handle: Long, capacity: Long)

    external fun nativeAdd(handle: Long, key: Long, vector: FloatArray)

    external fun nativeSearch(
        handle: Long,
        query: FloatArray,
        count: Int,
        keysOut: LongArray,
        distancesOut: FloatArray
    ): Int

    external fun nativeSize(handle: Long): Long

    external fun nativeMemoryUsage(handle: Long): Long

    external fun nativeSave(handle: Long, path: String)

    external fun nativeFree(handle: Long)
}
