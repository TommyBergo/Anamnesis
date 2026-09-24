package com.example.executorchllamademo.rag.repository

import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.rag.data.ClinicalChunk_
import com.example.executorchllamademo.rag.data.ScoredChunk
import io.objectbox.Box
import io.objectbox.BoxStore
import io.objectbox.query.QueryBuilder
import kotlin.math.ln
import kotlin.math.sqrt

/** Combines dense semantic similarity (as in [MedicalVectorRepository]) and lexical TF-IDF, scored per patient, into a single tunable-weight hybrid ranking. */
class HybridVectorRepository(boxStore: BoxStore) {

    private val chunkBox: Box<ClinicalChunk> = boxStore.boxFor(ClinicalChunk::class.java)

    fun insertChunks(chunks: List<ClinicalChunk>) {
        chunkBox.put(chunks)
    }

    fun searchHybrid(
        queryText: String,
        queryEmbedding: FloatArray,
        patientId: String,
        topK: Int = 4,
        semanticWeight: Double = 0.5
    ): List<ScoredChunk> {
        val candidates = chunkBox.query()
            .equal(ClinicalChunk_.patientId, patientId, QueryBuilder.StringOrder.CASE_SENSITIVE)
            .build()
            .find()
        if (candidates.isEmpty()) return emptyList()

        val docTokenLists = candidates.map { tokenize(it.content) }
        val n = candidates.size
        val documentFrequency = HashMap<String, Int>()
        for (tokens in docTokenLists) {
            for (term in tokens.toSet()) documentFrequency[term] = (documentFrequency[term] ?: 0) + 1
        }
        fun idf(term: String): Double = ln(n.toDouble() / (1.0 + (documentFrequency[term] ?: 0))) + 1.0

        fun tfIdfVector(tokens: List<String>): Map<String, Double> {
            if (tokens.isEmpty()) return emptyMap()
            val termFrequency = HashMap<String, Int>()
            for (t in tokens) termFrequency[t] = (termFrequency[t] ?: 0) + 1
            val maxTf = termFrequency.values.maxOrNull() ?: 1
            return termFrequency.mapValues { (term, count) -> (count.toDouble() / maxTf) * idf(term) }
        }

        fun cosineLexical(a: Map<String, Double>, b: Map<String, Double>): Double {
            if (a.isEmpty() || b.isEmpty()) return 0.0
            var dot = 0.0
            for ((term, weightA) in a) {
                val weightB = b[term] ?: continue
                dot += weightA * weightB
            }
            val normA = sqrt(a.values.sumOf { it * it })
            val normB = sqrt(b.values.sumOf { it * it })
            if (normA == 0.0 || normB == 0.0) return 0.0
            return dot / (normA * normB)
        }

        val queryTfIdf = tfIdfVector(tokenize(queryText))

        fun cosineSemantic(a: FloatArray, b: FloatArray): Double {
            if (a.size != b.size || a.isEmpty()) return 0.0
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

        return candidates.mapIndexed { i, chunk ->
            val lexicalScore = cosineLexical(queryTfIdf, tfIdfVector(docTokenLists[i]))
            val semanticScore = chunk.embedding?.let { cosineSemantic(queryEmbedding, it) } ?: 0.0
            val hybridScore = semanticWeight * semanticScore + (1 - semanticWeight) * lexicalScore
            ScoredChunk(chunk = chunk, score = (hybridScore * 100).toFloat())
        }
            .sortedByDescending { it.score }
            .take(topK)
    }

    fun clearChunksForDocument(patientId: String, documentId: String) {
        chunkBox.query()
            .equal(ClinicalChunk_.patientId, patientId, QueryBuilder.StringOrder.CASE_SENSITIVE)
            .equal(ClinicalChunk_.documentId, documentId, QueryBuilder.StringOrder.CASE_SENSITIVE)
            .build()
            .remove()
    }

    companion object {
        private val TOKEN_REGEX = Regex("[a-zA-Z0-9]+")

        private fun tokenize(text: String): List<String> =
            TOKEN_REGEX.findAll(text.lowercase()).map { it.value }.filter { it.length > 2 }.toList()
    }
}
