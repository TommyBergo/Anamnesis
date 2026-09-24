package com.example.executorchllamademo.rag

import android.util.Log
import com.example.executorchllamademo.rag.embedding.Embedder
import com.example.executorchllamademo.rag.repository.HybridBm25MediaPipeRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

/** On-device hybrid retriever that fuses BM25 and dense-embedder rankings by reciprocal rank fusion (RRF), per the Anamnesis paper's §4.3 Eq. 1. */
class RrfBm25MediaPipeRagEngine(
    private val vectorRepository: HybridBm25MediaPipeRepository,
    private val embedder: Embedder
) {

    suspend fun preparePrompt(
        userQuery: String,
        patientId: String,
        patientBackgroundSummary: String? = null,
        recentHistory: String? = null,
        topK: Int = 2,
        maxContextCharsPerChunk: Int = 400
    ): RagOutput = withContext(Dispatchers.Default) {

        val queryVector = embedder.generateEmbedding(userQuery, isQuery = true)

        val rawScores = vectorRepository.computeRawScores(
            queryText = userQuery,
            queryEmbedding = queryVector,
            patientId = patientId
        )
        val sortedChunks = HybridBm25MediaPipeRepository.fuseRrf(rawScores, topK)
        val relevancePercentages = HybridBm25MediaPipeRepository.relevancePercentages(rawScores)

        Log.i(
            TAG,
            "[RETRIEVAL] requestedTopK=$topK candidatePoolSize=${rawScores.candidates.size} " +
                "matches=${sortedChunks.size} scores=" +
                sortedChunks.joinToString(prefix = "[", postfix = "]") {
                    String.format(Locale.ROOT, "%.3f", it.score)
                }
        )

        val boundedChunks = sortedChunks.map { scored ->
            val content = scored.chunk.content
            if (content.length <= maxContextCharsPerChunk) {
                scored
            } else {
                val truncated = content.take(maxContextCharsPerChunk) + "..."
                scored.copy(chunk = scored.chunk.copy(content = truncated))
            }
        }

        val headerBuilder = StringBuilder()
        if (boundedChunks.isNotEmpty()) {
            headerBuilder.append("**Top ${boundedChunks.size} Relevant Chunks Extracted (RRF BM25+Dense, c=60):**\n")
            boundedChunks.forEachIndexed { index, scored ->
                val chunk = scored.chunk
                val relevance = relevancePercentages[chunk.id] ?: 0f
                val scoreFormatted = String.format(Locale.ROOT, "%.1f%%", relevance)
                val dateInfo = if (chunk.dateString.isNotBlank()) "${chunk.dateString} - " else ""
                headerBuilder.append("> **[Chunk #${index + 1}]** Coherence: **$scoreFormatted** | *${dateInfo}${chunk.documentType}*\n")
                headerBuilder.append("> \"${chunk.content.take(180).replace("\n", " ")}...\"\n\n")
            }
            headerBuilder.append("---\n\n")
        } else {
            headerBuilder.append("_No relevant clinical chunk found for this query._\n\n")
        }

        val contextTexts = boundedChunks.map {
            "[Notes from ${it.chunk.dateString} - ${it.chunk.documentType}]: ${it.chunk.content}"
        }

        val finalPrompt = MedicalPromptBuilder.buildConversationalRagPrompt(
            userMessage = userQuery,
            retrievedContexts = contextTexts,
            patientContext = patientBackgroundSummary,
            recentConversation = recentHistory
        )

        return@withContext RagOutput(
            prompt = finalPrompt,
            chunkHeaderMarkdown = headerBuilder.toString(),
            retrievedChunks = boundedChunks
        )
    }

    companion object {
        private const val TAG = "RrfBm25MediaPipeRagEngine"
    }
}
