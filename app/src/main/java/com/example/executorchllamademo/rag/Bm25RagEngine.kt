package com.example.executorchllamademo.rag

import android.util.Log
import com.example.executorchllamademo.rag.repository.Bm25VectorRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

/** Retrieval-and-prompt-assembly engine using Okapi BM25 lexical retrieval over [Bm25VectorRepository], producing the same [RagOutput] contract as the other RAG engines. */
class Bm25RagEngine(
    private val vectorRepository: Bm25VectorRepository
) {

    suspend fun preparePrompt(
        userQuery: String,
        patientId: String,
        patientBackgroundSummary: String? = null,
        recentHistory: String? = null,
        topK: Int = 2,
        maxContextCharsPerChunk: Int = 400
    ): RagOutput = withContext(Dispatchers.Default) {

        val sortedChunks = vectorRepository.searchSimilarChunks(
            queryText = userQuery,
            patientId = patientId,
            topK = topK
        ).sortedByDescending { it.score }

        Log.i(
            TAG,
            "[RETRIEVAL] requestedTopK=$topK matches=${sortedChunks.size} scores=" +
                sortedChunks.joinToString(prefix = "[", postfix = "]") {
                    String.format(Locale.ROOT, "%.2f", it.score)
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
            headerBuilder.append("**Top ${boundedChunks.size} Relevant Chunks Extracted (BM25):**\n")
            boundedChunks.forEachIndexed { index, scored ->
                val chunk = scored.chunk
                val scoreFormatted = String.format(Locale.ROOT, "%.2f", scored.score)
                val dateInfo = if (chunk.dateString.isNotBlank()) "${chunk.dateString} - " else ""
                headerBuilder.append("> **[Chunk #${index + 1}]** BM25: **$scoreFormatted** | *${dateInfo}${chunk.documentType}*\n")
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
        private const val TAG = "Bm25RagEngine"
    }
}
