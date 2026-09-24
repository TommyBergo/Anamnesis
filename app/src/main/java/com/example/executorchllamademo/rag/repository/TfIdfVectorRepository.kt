package com.example.executorchllamademo.rag.repository

import android.content.Context
import android.util.Log
import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.rag.data.ScoredChunk
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.ln
import kotlin.math.sqrt

/** JSON-file-backed repository for the paper's standalone TF-IDF retrieval configuration (Table 3's "TF-IDF" row), computing cosine similarity in pure Kotlin per patient. */
class TfIdfVectorRepository(context: Context) {

    private val storeFile = File(context.filesDir, "tfidf_rag_chunks.json")
    private val chunks = mutableListOf<ClinicalChunk>()

    init {
        loadFromDisk()
    }

    @Synchronized
    fun insertChunks(newChunks: List<ClinicalChunk>) {
        chunks.addAll(newChunks)
        persistToDisk()
    }

    @Synchronized
    fun clearAll() {
        chunks.clear()
        persistToDisk()
    }

    @Synchronized
    fun searchSimilarChunks(queryText: String, patientId: String, topK: Int): List<ScoredChunk> {
        val candidates = chunks.filter { it.patientId == patientId }
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

        fun cosine(a: Map<String, Double>, b: Map<String, Double>): Double {
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

        val queryVector = tfIdfVector(tokenize(queryText))
        return candidates.mapIndexed { i, chunk -> chunk to tfIdfVector(docTokenLists[i]) }
            .map { (chunk, vec) -> ScoredChunk(chunk = chunk, score = (cosine(queryVector, vec) * 100).toFloat()) }
            .sortedByDescending { it.score }
            .take(topK)
    }

    private fun persistToDisk() {
        try {
            val arr = JSONArray()
            for (c in chunks) {
                arr.put(
                    JSONObject().apply {
                        put("patientId", c.patientId)
                        put("documentType", c.documentType)
                        put("documentId", c.documentId)
                        put("content", c.content)
                        put("dateString", c.dateString)
                        put("timestamp", c.timestamp)
                    }
                )
            }
            storeFile.writeText(arr.toString())
        } catch (e: Exception) {
            Log.e(TAG, "Failed to persist chunks to ${storeFile.path}: ${e.message}", e)
        }
    }

    private fun loadFromDisk() {
        if (!storeFile.exists()) return
        try {
            val arr = JSONArray(storeFile.readText())
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                chunks.add(
                    ClinicalChunk(
                        patientId = obj.getString("patientId"),
                        documentType = obj.optString("documentType", ""),
                        documentId = obj.optString("documentId", ""),
                        content = obj.getString("content"),
                        dateString = obj.optString("dateString", ""),
                        timestamp = obj.optLong("timestamp", System.currentTimeMillis())
                    )
                )
            }
            Log.i(TAG, "Loaded ${chunks.size} chunks from ${storeFile.path}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load chunks from ${storeFile.path}: ${e.message}", e)
        }
    }

    companion object {
        private const val TAG = "TfIdfVectorRepository"
        private val TOKEN_REGEX = Regex("[a-zA-Z0-9]+")

        private fun tokenize(text: String): List<String> =
            TOKEN_REGEX.findAll(text.lowercase()).map { it.value }.filter { it.length > 2 }.toList()
    }
}
