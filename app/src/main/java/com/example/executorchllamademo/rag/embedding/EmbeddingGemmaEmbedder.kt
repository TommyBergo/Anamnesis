package com.example.executorchllamademo.rag.embedding

import android.content.Context
import com.google.mediapipe.framework.MediaPipeException
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder
import java.io.Closeable

/** Thrown when EmbeddingGemma embedding generation or initialization fails. */
class EmbeddingGemmaEmbedderException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/** Embeds text with Google's EmbeddingGemma model (Table 3's "EmbeddingGemma" row) via the MediaPipe Tasks `TextEmbedder` API, retrying with shrunk input on sequence-length overflow. */
class EmbeddingGemmaEmbedder(
    private val context: Context,
    private val modelName: String = "embedding_gemma_mp.task",
    private val outputDimension: Int = 768
) : Embedder, Closeable {

    override val backendId = "EmbeddingGemmaEmbedder:embeddinggemma-300m-seq512"
    override val embeddingDimension: Int get() = outputDimension

    companion object {
        private const val QUERY_PREFIX = "task: search result | query: "
        private const val DOCUMENT_PREFIX = "title: none | text: "

        private const val MAX_TRUNCATION_ATTEMPTS = 8
        private const val TRUNCATION_SHRINK_FACTOR = 0.5
    }

    private fun createTextEmbedder(): TextEmbedder {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(modelName)
            .build()
        val options = TextEmbedder.TextEmbedderOptions.builder()
            .setBaseOptions(baseOptions)
            .build()
        return TextEmbedder.createFromOptions(context, options)
    }

    @Volatile
    private var textEmbedder: TextEmbedder? = try {
        createTextEmbedder()
    } catch (e: Exception) {
        throw EmbeddingGemmaEmbedderException("Failed to initialize MediaPipe TextEmbedder from '$modelName': ${e.message}", e)
    }

    @Synchronized
    override fun generateEmbedding(text: String, isQuery: Boolean): FloatArray {
        val prefix = if (isQuery) QUERY_PREFIX else DOCUMENT_PREFIX
        var candidateText = text
        var lastError: MediaPipeException? = null

        repeat(MAX_TRUNCATION_ATTEMPTS) {
            val embedder = textEmbedder
                ?: throw EmbeddingGemmaEmbedderException("EmbeddingGemmaEmbedder has already been closed")

            val floatEmbedding = try {
                embedder.embed(prefix + candidateText).embeddingResult().embeddings().first().floatEmbedding()
            } catch (e: MediaPipeException) {
                lastError = e
                try { embedder.close() } catch (_: Exception) { }
                textEmbedder = createTextEmbedder()
                candidateText = candidateText.take((candidateText.length * TRUNCATION_SHRINK_FACTOR).toInt())
                return@repeat
            } catch (e: Exception) {
                throw EmbeddingGemmaEmbedderException("MediaPipe TextEmbedder inference failed: ${e.message}", e)
            }

            if (floatEmbedding.size != outputDimension) {
                throw EmbeddingGemmaEmbedderException(
                    "Unexpected embedding dimension: got ${floatEmbedding.size}, expected $outputDimension"
                )
            }
            return floatEmbedding
        }

        throw EmbeddingGemmaEmbedderException(
            "MediaPipe TextEmbedder inference failed after $MAX_TRUNCATION_ATTEMPTS truncation attempts: ${lastError?.message}",
            lastError
        )
    }

    @Synchronized
    override fun close() {
        textEmbedder?.close()
        textEmbedder = null
    }
}
