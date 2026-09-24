package com.example.executorchllamademo.rag

import android.util.Log
import com.example.executorchllamademo.rag.embedding.Embedder
import com.example.executorchllamademo.rag.repository.HybridVectorRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

/** Combines semantic retrieval (as in [MedicalRagEngine]) with lexical TF-IDF via [HybridVectorRepository] into a single weighted-blend retrieval path. */
class HybridRagEngine(
    private val vectorRepository: HybridVectorRepository,
    private val embedder: Embedder,
    private val semanticWeight: Double = 0.5
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

        val sortedChunks = vectorRepository.searchHybrid(
            queryText = userQuery,
            queryEmbedding = queryVector,
            patientId = patientId,
            topK = topK,
            semanticWeight = semanticWeight
        ).sortedByDescending { it.score }

        Log.i(
            TAG,
            "[RETRIEVAL] requestedTopK=$topK semanticWeight=$semanticWeight matches=${sortedChunks.size} scores=" +
                sortedChunks.joinToString(prefix = "[", postfix = "]") {
                    String.format(Locale.ROOT, "%.1f%%", it.score)
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
            headerBuilder.append("**Top ${boundedChunks.size} Relevant Chunks Extracted (Hybrid):**\n")
            boundedChunks.forEachIndexed { index, scored ->
                val chunk = scored.chunk
                val scoreFormatted = String.format(Locale.ROOT, "%.1f%%", scored.score)
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
        private const val TAG = "HybridRagEngine"
    }
}
