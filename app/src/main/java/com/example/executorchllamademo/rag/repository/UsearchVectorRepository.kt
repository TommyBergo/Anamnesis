package com.example.executorchllamademo.rag.repository

import android.content.Context
import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.rag.data.ScoredChunk
import java.io.File

/** Vector-store benchmark candidate using USearch's HNSW index via [UsearchNative] (Table 10's "USearch (JNI)" row). */
class UsearchVectorRepository(context: Context, private val dimensions: Int = 768) {

    private val storeFile = File(context.filesDir, "usearch_vector_benchmark.bin")
    private val chunksByKey = mutableListOf<ClinicalChunk>()
    private var handle: Long = UsearchNative.nativeInit(dimensions)

    init {
        storeFile.delete()
    }

    fun insertChunks(newChunks: List<ClinicalChunk>) {
        val eligible = newChunks.filter { it.embedding?.size == dimensions }
        if (eligible.isEmpty()) return
        UsearchNative.nativeReserve(handle, (chunksByKey.size + eligible.size).toLong())
        for (chunk in eligible) {
            val key = chunksByKey.size.toLong()
            UsearchNative.nativeAdd(handle, key, chunk.embedding!!)
            chunksByKey.add(chunk)
        }
    }

    fun searchTopK(queryEmbedding: FloatArray, topK: Int): List<ScoredChunk> {
        if (chunksByKey.isEmpty()) return emptyList()
        val keys = LongArray(topK)
        val distances = FloatArray(topK)
        val found = UsearchNative.nativeSearch(handle, queryEmbedding, topK, keys, distances)
        return (0 until found).map { i ->
            val chunk = chunksByKey[keys[i].toInt()]
            val similarity = 1.0 - (distances[i].toDouble() / 2.0)
            ScoredChunk(chunk = chunk, score = (similarity * 100).toFloat())
        }.sortedByDescending { it.score }
    }

    fun diskSizeBytes(): Long {
        if (chunksByKey.isEmpty()) return 0L
        UsearchNative.nativeSave(handle, storeFile.absolutePath)
        return storeFile.length()
    }

    fun close() {
        UsearchNative.nativeFree(handle)
        handle = 0L
        storeFile.delete()
        chunksByKey.clear()
    }
}
