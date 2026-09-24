package com.example.executorchllamademo.rag

import kotlin.math.ln

/** Shared, per-query ranked-retrieval metrics (Hit@k, Precision@k, MRR, MAP, nDCG@k) used by every RAG evaluation test in this package. */
object RetrievalMetrics {

    fun hitAtK(relevance: List<Boolean>, k: Int): Boolean = relevance.take(k).any { it }

    fun precisionAtK(relevance: List<Boolean>, k: Int): Double {
        if (k <= 0) return 0.0
        return relevance.take(k).count { it }.toDouble() / k
    }

    fun reciprocalRank(relevance: List<Boolean>): Double {
        val firstHitIndex = relevance.indexOfFirst { it }
        return if (firstHitIndex == -1) 0.0 else 1.0 / (firstHitIndex + 1)
    }

    fun averagePrecision(relevance: List<Boolean>): Double {
        var relevantSoFar = 0
        var precisionSum = 0.0
        for ((index, isRelevant) in relevance.withIndex()) {
            if (isRelevant) {
                relevantSoFar++
                precisionSum += relevantSoFar.toDouble() / (index + 1)
            }
        }
        return if (relevantSoFar == 0) 0.0 else precisionSum / relevantSoFar
    }

    fun ndcgAt(relevance: List<Boolean>, k: Int): Double {
        if (k <= 0) return 0.0
        fun discountAt(index: Int) = ln(index + 2.0) / ln(2.0)

        val dcg = relevance.take(k).withIndex().sumOf { (index, isRelevant) ->
            if (isRelevant) 1.0 / discountAt(index) else 0.0
        }
        val idealRelevantCount = relevance.count { it }.coerceAtMost(k)
        if (idealRelevantCount == 0) return 0.0
        val idcg = (0 until idealRelevantCount).sumOf { index -> 1.0 / discountAt(index) }
        return if (idcg == 0.0) 0.0 else dcg / idcg
    }
}
