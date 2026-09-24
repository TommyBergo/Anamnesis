package com.example.executorchllamademo.rag.embedding

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/** Thrown when multilingual-e5-small embedding generation or initialization fails. */
class MultilingualE5SmallEmbedderException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/** Embeds text with the multilingual-e5-small LiteRT model (Table 3's "E5 multilingual small" row) via [XlmRobertaUnigramTokenizer] and E5's required query:/passage: prefix. */
class MultilingualE5SmallEmbedder(
    context: Context,
    modelName: String = "e5_small_ml.tflite",
    tokenizerFileName: String = "e5_small_ml_tokenizer.json",
    private val sequenceLength: Int = 512,
    private val outputDimension: Int = 384
) : Embedder, Closeable {

    override val backendId = "MultilingualE5SmallEmbedder:multilingual-e5-small"
    override val embeddingDimension: Int get() = outputDimension

    private val tokenizer: XlmRobertaUnigramTokenizer = try {
        XlmRobertaUnigramTokenizer(context, tokenizerFileName)
    } catch (e: XlmRobertaUnigramTokenizerException) {
        throw e
    } catch (e: Exception) {
        throw MultilingualE5SmallEmbedderException(
            "Failed to initialize multilingual-e5-small tokenizer from '$tokenizerFileName'", e
        )
    }

    @Volatile
    private var interpreter: Interpreter? = try {
        val modelBuffer = loadModelFile(context, modelName)
        val options = Interpreter.Options().apply {
            numThreads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        }
        Interpreter(modelBuffer, options)
    } catch (e: Exception) {
        throw MultilingualE5SmallEmbedderException("Failed to initialize TFLite model '$modelName': ${e.message}", e)
    }

    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val externalFile = File(EXTERNAL_MODEL_DIR, modelPath)
        if (externalFile.exists()) {
            return FileInputStream(externalFile).use { inputStream ->
                inputStream.channel.use { fileChannel ->
                    fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, externalFile.length())
                }
            }
        }

        val assetFd = try {
            context.assets.openFd(modelPath)
        } catch (e: IOException) {
            throw MultilingualE5SmallEmbedderException(
                "Embedding model not found at '$externalFile' or as asset '$modelPath'", e
            )
        }
        return FileInputStream(assetFd.fileDescriptor).use { inputStream ->
            inputStream.channel.use { fileChannel ->
                fileChannel.map(
                    FileChannel.MapMode.READ_ONLY,
                    assetFd.startOffset,
                    assetFd.declaredLength
                )
            }
        }
    }

    @Synchronized
    override fun generateEmbedding(text: String, isQuery: Boolean): FloatArray {
        val safeInterpreter = interpreter
            ?: throw MultilingualE5SmallEmbedderException("MultilingualE5SmallEmbedder has already been closed")

        val prefixedText = if (isQuery) QUERY_PREFIX + text else PASSAGE_PREFIX + text
        val encoding = tokenizer.encode(prefixedText, sequenceLength)
        val inputIds = arrayOf(encoding.ids)
        val attentionMask = arrayOf(encoding.attentionMask)
        val outputEmbedding = Array(1) { FloatArray(outputDimension) }

        try {
            safeInterpreter.runForMultipleInputsOutputs(
                arrayOf<Any>(inputIds, attentionMask),
                mutableMapOf<Int, Any>(0 to outputEmbedding)
            )
        } catch (e: Exception) {
            throw MultilingualE5SmallEmbedderException("TFLite inference failed while generating embedding: ${e.message}", e)
        }

        return outputEmbedding[0]
    }

    @Synchronized
    override fun close() {
        interpreter?.close()
        interpreter = null
    }

    companion object {
        private const val EXTERNAL_MODEL_DIR = "/data/local/tmp/llama"
        private const val QUERY_PREFIX = "query: "
        private const val PASSAGE_PREFIX = "passage: "
    }
}
