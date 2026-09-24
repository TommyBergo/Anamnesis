package com.example.executorchllamademo.rag.embedding

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/** Thrown when Granite embedding generation or initialization fails. */
class GraniteEmbedderException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/** Embeds text with IBM's Granite-embedding-311m-multilingual-r2 LiteRT model (Table 3's "Granite" row) via TFLite's named-signature API and [GraniteTokenizer]. */
class GraniteEmbedder(
    context: Context,
    modelName: String = "granite_embedding_311m_multilingual.tflite",
    tokenizerFileName: String = "granite_tokenizer.json",
    private val sequenceLength: Int = 512,
    private val outputDimension: Int = 768
) : Embedder, Closeable {

    override val backendId = "GraniteEmbedder:granite-embedding-311m-multilingual-r2"
    override val embeddingDimension: Int get() = outputDimension

    companion object {
        private const val SIGNATURE_KEY = "embed_512"
        private const val INPUT_IDS_NAME = "input_ids"
        private const val ATTENTION_MASK_NAME = "attention_mask"
        private const val OUTPUT_NAME = "output_0"
    }

    private val tokenizer: GraniteTokenizer = try {
        GraniteTokenizer(context, tokenizerFileName)
    } catch (e: GraniteTokenizerException) {
        throw e
    } catch (e: Exception) {
        throw GraniteEmbedderException("Failed to initialize Granite tokenizer from '$tokenizerFileName'", e)
    }

    @Volatile
    private var interpreter: Interpreter? = try {
        val modelBuffer = loadModelFile(context, modelName)
        val options = Interpreter.Options().apply {
            numThreads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        }
        Interpreter(modelBuffer, options)
    } catch (e: Exception) {
        throw GraniteEmbedderException("Failed to initialize TFLite model '$modelName': ${e.message}", e)
    }

    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val assetFd = try {
            context.assets.openFd(modelPath)
        } catch (e: IOException) {
            throw GraniteEmbedderException("Embedding model asset '$modelPath' not found", e)
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
            ?: throw GraniteEmbedderException("GraniteEmbedder has already been closed")

        val encoding = tokenizer.encode(text, sequenceLength)
        val inputs = mapOf(
            INPUT_IDS_NAME to arrayOf(encoding.ids),
            ATTENTION_MASK_NAME to arrayOf(encoding.attentionMask)
        )
        val outputEmbedding = Array(1) { FloatArray(outputDimension) }
        val outputs = mutableMapOf<String, Any>(OUTPUT_NAME to outputEmbedding)

        try {
            safeInterpreter.runSignature(inputs, outputs, SIGNATURE_KEY)
        } catch (e: Exception) {
            throw GraniteEmbedderException("TFLite inference failed while generating embedding: ${e.message}", e)
        }

        return outputEmbedding[0]
    }

    @Synchronized
    override fun close() {
        interpreter?.close()
        interpreter = null
    }
}
