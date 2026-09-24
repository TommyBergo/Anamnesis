package com.example.executorchllamademo.rag

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.executorchllamademo.rag.embedding.GraniteEmbedder
import com.example.executorchllamademo.rag.embedding.embeddingFingerprint
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.sqrt

/** Standalone, on-device sanity check for [GraniteEmbedder], confirming basic correctness plus cross-lingual (English/Italian) semantic signal since this app's corpus is multilingual. */
@RunWith(AndroidJUnit4::class)
class GraniteEmbedderTest {

    companion object {
        private const val TAG = "GraniteEmbedderTest"

        private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
            var dot = 0f; var normA = 0f; var normB = 0f
            for (i in a.indices) {
                dot += a[i] * b[i]
                normA += a[i] * a[i]
                normB += b[i] * b[i]
            }
            return dot / (sqrt(normA) * sqrt(normB))
        }
    }

    @Test
    fun testGraniteEmbedderProducesSaneEmbeddings() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val embedder = GraniteEmbedder(context)

        try {
            val query = embedder.generateEmbedding(
                "What was the result of the cardiac catheterization?", isQuery = true
            )
            val relatedDoc = embedder.generateEmbedding(
                "Cardiac catheterization revealed a 100% occlusion of the right coronary artery.",
                isQuery = false
            )
            val unrelatedDoc = embedder.generateEmbedding(
                "The patient enjoys gardening and reading on weekends.", isQuery = false
            )

            Log.i(TAG, "query: ${query.embeddingFingerprint()}")
            Log.i(TAG, "relatedDoc: ${relatedDoc.embeddingFingerprint()}")
            Log.i(TAG, "unrelatedDoc: ${unrelatedDoc.embeddingFingerprint()}")

            assertEquals(768, query.size)
            assertFalse("Embedding must not contain NaN", query.any { it.isNaN() })
            assertFalse("Embedding must not contain Inf", query.any { it.isInfinite() })

            val simRelated = cosineSimilarity(query, relatedDoc)
            val simUnrelated = cosineSimilarity(query, unrelatedDoc)
            Log.i(TAG, "cosine(query, relatedDoc)=$simRelated cosine(query, unrelatedDoc)=$simUnrelated")

            assertTrue(
                "Expected the related document to score higher than the unrelated one " +
                    "(related=$simRelated, unrelated=$simUnrelated)",
                simRelated > simUnrelated
            )
        } finally {
            embedder.close()
        }
    }

    @Test
    fun testGraniteEmbedderCrossLingualEnglishItalian() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val embedder = GraniteEmbedder(context)

        try {
            val english = embedder.generateEmbedding(
                "The patient was admitted for chest pain and shortness of breath.", isQuery = false
            )
            val italianParaphrase = embedder.generateEmbedding(
                "Il paziente e stato ricoverato per dolore toracico e difficolta respiratoria.",
                isQuery = false
            )
            val italianUnrelated = embedder.generateEmbedding(
                "Al paziente piace fare giardinaggio nei fine settimana.", isQuery = false
            )

            val simParaphrase = cosineSimilarity(english, italianParaphrase)
            val simUnrelated = cosineSimilarity(english, italianUnrelated)
            Log.i(
                TAG,
                "cross-lingual cosine(en, it-paraphrase)=$simParaphrase cosine(en, it-unrelated)=$simUnrelated"
            )

            assertTrue(
                "Expected the Italian paraphrase to score higher than unrelated Italian text " +
                    "(paraphrase=$simParaphrase, unrelated=$simUnrelated) - this is the whole " +
                    "point of using a multilingual embedder for this app's mixed-language corpus",
                simParaphrase > simUnrelated
            )
        } finally {
            embedder.close()
        }
    }
}
