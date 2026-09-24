package com.example.executorchllamademo.rag.repository

import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.rag.data.ClinicalChunk_
import com.example.executorchllamademo.rag.data.ScoredChunk
import io.objectbox.Box
import io.objectbox.BoxStore
import io.objectbox.query.QueryBuilder
import kotlin.math.sqrt

/** Dense-embedding repository over ObjectBox, ranking a patient's chunks by cosine similarity, or via the global HNSW index when no patient is specified. */
class MedicalVectorRepository(boxStore: BoxStore) {

    private val chunkBox: Box<ClinicalChunk> = boxStore.boxFor(ClinicalChunk::class.java)

    fun insertChunks(chunks: List<ClinicalChunk>) {
        chunkBox.put(chunks)
    }

    fun searchSimilarChunksWithScore(
        queryEmbedding: FloatArray,
        patientId: String? = null,
        topK: Int = 4
    ): List<ScoredChunk> {
        if (!patientId.isNullOrBlank()) {
            val candidates = chunkBox.query()
                .equal(ClinicalChunk_.patientId, patientId, QueryBuilder.StringOrder.CASE_SENSITIVE)
                .build()
                .find()

            return candidates
                .mapNotNull { chunk ->
                    val embedding = chunk.embedding ?: return@mapNotNull null
                    ScoredChunk(chunk = chunk, score = cosineSimilarity(queryEmbedding, embedding) * 100f)
                }
                .sortedByDescending { it.score }
                .take(topK)
        }

        val resultsWithScores = chunkBox.query()
            .nearestNeighbors(ClinicalChunk_.embedding, queryEmbedding, topK)
            .build()
            .findWithScores()

        return resultsWithScores.map { objectWithScore ->
            val distance = objectWithScore.score.toFloat()
            val similarityPercentage = (1.0f / (1.0f + distance)) * 100f

            ScoredChunk(
                chunk = objectWithScore.get(),
                score = similarityPercentage
            )
        }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size || a.isEmpty()) return 0f
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        if (normA == 0f || normB == 0f) return 0f
        return dot / (sqrt(normA) * sqrt(normB))
    }

    fun clearChunksForDocument(patientId: String, documentId: String) {
        chunkBox.query()
            .equal(ClinicalChunk_.patientId, patientId, QueryBuilder.StringOrder.CASE_SENSITIVE)
            .equal(ClinicalChunk_.documentId, documentId, QueryBuilder.StringOrder.CASE_SENSITIVE)
            .build()
            .remove()
    }
}