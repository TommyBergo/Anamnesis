package com.example.executorchllamademo.rag.repository

import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.rag.data.ClinicalChunk_
import com.example.executorchllamademo.rag.data.ScoredChunk
import io.objectbox.Box
import io.objectbox.BoxStore
import io.objectbox.query.QueryBuilder
import kotlin.math.ln
import kotlin.math.sqrt

/** Holds the raw, unfused per-candidate BM25 and dense-embedder scores for one query, computed once so multiple fusion strategies can reuse them. */
data class RawHybridCandidateScores(
    val candidates: List<ClinicalChunk>,
    val bm25Raw: List<Double>,
    val denseRaw: List<Double>
)

/** Computes raw per-candidate BM25 and dense-embedder scores for a patient's chunks, then fuses them by weighted blend ([fuse]) or reciprocal rank fusion ([fuseRrf]). */
class HybridBm25MediaPipeRepository(boxStore: BoxStore) {

    private val chunkBox: Box<ClinicalChunk> = boxStore.boxFor(ClinicalChunk::class.java)

    fun insertChunks(chunks: List<ClinicalChunk>) {
        chunkBox.put(chunks)
    }

    fun computeRawScores(
        queryText: String,
        queryEmbedding: FloatArray,
        patientId: String
    ): RawHybridCandidateScores {
        val candidates = chunkBox.query()
            .equal(ClinicalChunk_.patientId, patientId, QueryBuilder.StringOrder.CASE_SENSITIVE)
            .build()
            .find()
        if (candidates.isEmpty()) return RawHybridCandidateScores(emptyList(), emptyList(), emptyList())

        val docTokenLists = candidates.map { tokenize(it.content) }
        val n = candidates.size
        val avgDocLength = docTokenLists.sumOf { it.size }.toDouble() / n

        val documentFrequency = HashMap<String, Int>()
        for (tokens in docTokenLists) {
            for (term in tokens.toSet()) documentFrequency[term] = (documentFrequency[term] ?: 0) + 1
        }
        fun bm25Idf(term: String): Double {
            val nt = documentFrequency[term] ?: 0
            return ln((n - nt + 0.5) / (nt + 0.5) + 1.0)
        }

        val queryTerms = tokenize(queryText)
        fun bm25Score(docTokens: List<String>): Double {
            if (docTokens.isEmpty() || queryTerms.isEmpty()) return 0.0
            val termFrequency = HashMap<String, Int>()
            for (t in docTokens) termFrequency[t] = (termFrequency[t] ?: 0) + 1
            val docLength = docTokens.size.toDouble()
            var score = 0.0
            for (term in queryTerms.toSet()) {
                val tf = termFrequency[term] ?: continue
                val numerator = tf * (BM25_K1 + 1.0)
                val denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (docLength / avgDocLength))
                score += bm25Idf(term) * (numerator / denominator)
            }
            return score
        }

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

        val bm25Raw = docTokenLists.map { bm25Score(it) }
        val denseRaw = candidates.map { chunk ->
            chunk.embedding?.let { cosineSemantic(queryEmbedding, it) } ?: 0.0
        }
        return RawHybridCandidateScores(candidates, bm25Raw, denseRaw)
    }

    fun clearChunksForDocument(patientId: String, documentId: String) {
        chunkBox.query()
            .equal(ClinicalChunk_.patientId, patientId, QueryBuilder.StringOrder.CASE_SENSITIVE)
            .equal(ClinicalChunk_.documentId, documentId, QueryBuilder.StringOrder.CASE_SENSITIVE)
            .build()
            .remove()
    }

    companion object {
        private const val BM25_K1 = 1.2
        private const val BM25_B = 0.75
        private val TOKEN_REGEX = Regex("[a-zA-Z0-9]+")

        private fun tokenize(text: String): List<String> =
            TOKEN_REGEX.findAll(text.lowercase()).map { it.value }.filter { it.length > 2 }.toList()

        fun fuse(raw: RawHybridCandidateScores, alpha: Double, topK: Int): List<ScoredChunk> {
            if (raw.candidates.isEmpty()) return emptyList()
            val bm25Norm = minMaxNormalize(raw.bm25Raw)
            val denseNorm = minMaxNormalize(raw.denseRaw)
            return raw.candidates.indices
                .map { i ->
                    val hybridScore = alpha * denseNorm[i] + (1.0 - alpha) * bm25Norm[i]
                    ScoredChunk(chunk = raw.candidates[i], score = (hybridScore * 100).toFloat())
                }
                .sortedWith(compareByDescending<ScoredChunk> { it.score }.thenBy { it.chunk.id })
                .take(topK)
        }

        private fun minMaxNormalize(values: List<Double>): List<Double> {
            if (values.isEmpty()) return values
            val min = values.min()
            val max = values.max()
            val range = max - min
            return if (range > 0.0) values.map { (it - min) / range } else values.map { 0.0 }
        }

        fun relevancePercentages(raw: RawHybridCandidateScores): Map<Long, Float> {
            if (raw.candidates.isEmpty()) return emptyMap()
            val bm25Norm = minMaxNormalize(raw.bm25Raw)
            val denseNorm = minMaxNormalize(raw.denseRaw)
            return raw.candidates.indices.associate { i ->
                raw.candidates[i].id to (((bm25Norm[i] + denseNorm[i]) / 2.0) * 100).toFloat()
            }
        }

        private const val RRF_C = 60.0

        fun fuseRrf(raw: RawHybridCandidateScores, topK: Int): List<ScoredChunk> {
            if (raw.candidates.isEmpty()) return emptyList()
            val n = raw.candidates.size

            fun ranksOf(scores: List<Double>): IntArray {
                val order = (0 until n).sortedWith(
                    compareByDescending<Int> { scores[it] }.thenBy { raw.candidates[it].id }
                )
                val ranks = IntArray(n)
                for ((rank, candidateIndex) in order.withIndex()) ranks[candidateIndex] = rank + 1
                return ranks
            }

            val bm25Ranks = ranksOf(raw.bm25Raw)
            val denseRanks = ranksOf(raw.denseRaw)

            return raw.candidates.indices
                .map { i ->
                    val rrfScore = 1.0 / (RRF_C + bm25Ranks[i]) + 1.0 / (RRF_C + denseRanks[i])
                    ScoredChunk(chunk = raw.candidates[i], score = (rrfScore * 100).toFloat())
                }
                .sortedWith(compareByDescending<ScoredChunk> { it.score }.thenBy { it.chunk.id })
                .take(topK)
        }
    }
}
