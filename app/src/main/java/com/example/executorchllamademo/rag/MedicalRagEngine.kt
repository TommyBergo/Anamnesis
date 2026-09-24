package com.example.executorchllamademo.rag

import android.util.Log
import com.example.executorchllamademo.rag.data.ScoredChunk
import com.example.executorchllamademo.rag.embedding.Embedder
import com.example.executorchllamademo.rag.embedding.PureEmbedder
import com.example.executorchllamademo.rag.embedding.embeddingFingerprint
import com.example.executorchllamademo.rag.repository.MedicalVectorRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

/** Bundles the assembled RAG prompt together with its debug-header markdown and the chunks that were retrieved. */
data class RagOutput(
    val prompt: String,
    val chunkHeaderMarkdown: String,
    val retrievedChunks: List<ScoredChunk>
)

/** Retrieval-and-prompt-assembly engine using dense embedding similarity over [MedicalVectorRepository] to build the RAG prompt. */
class MedicalRagEngine(
    private val vectorRepository: MedicalVectorRepository,
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

        if (embedder is PureEmbedder) {
            Log.w(
                TAG,
                "[QUERY] embedder='${embedder.backendId}' is the PureEmbedder fallback (no real " +
                    "semantic embeddings). Skipping retrieval."
            )
            return@withContext RagOutput(
                prompt = MedicalPromptBuilder.buildConversationalRagPrompt(
                    userMessage = userQuery,
                    retrievedContexts = emptyList(),
                    patientContext = patientBackgroundSummary,
                    recentConversation = recentHistory
                ),
                chunkHeaderMarkdown = "_Clinical retrieval unavailable: active embedding backend is " +
                    "'${embedder.backendId}' (semantic embedding model failed to load)._\n\n",
                retrievedChunks = emptyList()
            )
        }

        val queryVector = embedder.generateEmbedding(userQuery, isQuery = true)
        Log.i(TAG, "[QUERY] embedder='${embedder.backendId}' ${queryVector.embeddingFingerprint()}")

        val rawScoredChunks = vectorRepository.searchSimilarChunksWithScore(
            queryEmbedding = queryVector,
            patientId = patientId,
            topK = topK
        )

        val sortedChunks = rawScoredChunks.sortedByDescending { it.score }
        Log.i(
            TAG,
            "[RETRIEVAL] requestedTopK=$topK matches=${sortedChunks.size} scores=" +
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
            headerBuilder.append("**Top ${boundedChunks.size} Relevant Chunks Extracted:**\n")
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
        private const val TAG = "MedicalRagEngine"
    }
}
