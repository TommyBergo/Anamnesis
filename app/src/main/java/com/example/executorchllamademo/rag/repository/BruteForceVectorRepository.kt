package com.example.executorchllamademo.rag.repository

import android.content.Context
import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.rag.data.ScoredChunk
import java.io.DataOutputStream
import java.io.File
import kotlin.math.sqrt

/** Vector-store benchmark baseline that ranks chunks by exact linear-scan cosine similarity, serving as ground truth for the other stores' recall@10. */
class BruteForceVectorRepository(context: Context, private val dimensions: Int = 768) {

    private val storeFile = File(context.filesDir, "bruteforce_vector_benchmark.bin")
    private val chunks = mutableListOf<ClinicalChunk>()

    init {
        storeFile.delete()
    }

    fun insertChunks(newChunks: List<ClinicalChunk>) {
        chunks.addAll(newChunks.filter { it.embedding?.size == dimensions })
        persistToDisk()
    }

    fun searchTopK(queryEmbedding: FloatArray, topK: Int): List<ScoredChunk> {
        return chunks.mapNotNull { chunk ->
            val embedding = chunk.embedding ?: return@mapNotNull null
            ScoredChunk(chunk = chunk, score = (cosineSimilarity(queryEmbedding, embedding) * 100).toFloat())
        }.sortedByDescending { it.score }.take(topK)
    }

    fun diskSizeBytes(): Long = storeFile.length()

    fun close() {
        storeFile.delete()
        chunks.clear()
    }

    private fun persistToDisk() {
        DataOutputStream(storeFile.outputStream().buffered()).use { out ->
            out.writeInt(chunks.size)
            for (chunk in chunks) {
                out.writeLong(chunk.id)
                out.writeUTF(chunk.patientId)
                out.writeUTF(chunk.documentType)
                out.writeUTF(chunk.documentId)
                out.writeUTF(chunk.content)
                out.writeUTF(chunk.dateString)
                out.writeLong(chunk.timestamp)
                val embedding = chunk.embedding!!
                out.writeInt(embedding.size)
                for (f in embedding) out.writeFloat(f)
            }
        }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Double {
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        if (normA == 0.0 || normB == 0.0) return 0.0
        return dot / (sqrt(normA) * sqrt(normB))
    }
}
