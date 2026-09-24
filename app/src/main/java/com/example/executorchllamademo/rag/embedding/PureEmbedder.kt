package com.example.executorchllamademo.rag.embedding

import kotlin.math.sqrt

/** Degraded, non-semantic fallback embedder that hashes words into a normalized vector when no real embedding model is available. */
class PureEmbedder : Embedder {

    override val backendId = "PureEmbedder:hash-fallback(MOCK, non-semantic)"
    override val embeddingDimension = 384

    override fun generateEmbedding(text: String, isQuery: Boolean): FloatArray {
        val dimensions = embeddingDimension
        val vector = FloatArray(dimensions)

        val words = text.lowercase()
            .split(Regex("\\W+"))
            .filter { it.isNotBlank() && it.length > 2 }

        if (words.isEmpty()) return vector

        for (word in words) {
            val hash = word.hashCode()
            val index = kotlin.math.abs(hash) % dimensions
            val sign = if (hash >= 0) 1f else -1f
            vector[index] += sign
        }

        var sumSquares = 0.0
        for (v in vector) {
            sumSquares += (v * v).toDouble()
        }
        val norm = sqrt(sumSquares).toFloat()

        if (norm > 0f) {
            for (i in vector.indices) {
                vector[i] /= norm
            }
        }

        return vector
    }
}