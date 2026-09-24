package com.example.executorchllamademo.rag.repository

import android.content.Context
import android.util.Log
import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.rag.data.ScoredChunk
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.ln

/** JSON-file-backed repository that scores each patient's own chunks with Okapi BM25 (k1=1.2, b=0.75), computed fresh per patient. */
class Bm25VectorRepository(context: Context) {

    private val storeFile = File(context.filesDir, "bm25_rag_chunks.json")
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
        val avgDocLength = docTokenLists.sumOf { it.size }.toDouble() / n

        val documentFrequency = HashMap<String, Int>()
        for (tokens in docTokenLists) {
            for (term in tokens.toSet()) documentFrequency[term] = (documentFrequency[term] ?: 0) + 1
        }
        fun idf(term: String): Double {
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
                val numerator = tf * (K1 + 1.0)
                val denominator = tf + K1 * (1.0 - B + B * (docLength / avgDocLength))
                score += idf(term) * (numerator / denominator)
            }
            return score
        }

        return candidates.mapIndexed { i, chunk -> chunk to bm25Score(docTokenLists[i]) }
            .map { (chunk, score) -> ScoredChunk(chunk = chunk, score = score.toFloat()) }
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
        private const val TAG = "Bm25VectorRepository"
        private const val K1 = 1.2
        private const val B = 0.75
        private val TOKEN_REGEX = Regex("[a-zA-Z0-9]+")

        private fun tokenize(text: String): List<String> =
            TOKEN_REGEX.findAll(text.lowercase()).map { it.value }.filter { it.length > 2 }.toList()
    }
}
