package com.example.executorchllamademo.rag.embedding

import java.util.Locale
import kotlin.math.sqrt

/** Produces embedding vectors for a text-embedding backend, distinguishing query-side from document-side text where the backend benefits from it. */
interface Embedder {
    val backendId: String

    val embeddingDimension: Int

    fun generateEmbedding(text: String, isQuery: Boolean = false): FloatArray
}

fun FloatArray.embeddingFingerprint(): String {
    var sumSquares = 0.0
    for (v in this) sumSquares += (v * v).toDouble()
    val norm = sqrt(sumSquares)
    val sample = take(4).joinToString(", ") { String.format(Locale.ROOT, "%.4f", it) }
    return "dim=$size l2norm=${String.format(Locale.ROOT, "%.4f", norm)} sample=[$sample]"
}
