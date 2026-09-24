package com.example.executorchllamademo.rag

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.executorchllamademo.rag.embedding.EmbeddingGemmaEmbedder
import com.example.executorchllamademo.rag.embedding.embeddingFingerprint
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.sqrt

/** Standalone, on-device sanity check for [EmbeddingGemmaEmbedder] in isolation, confirming embedding dimension/no-NaN and that semantically related text scores higher than unrelated text. */
@RunWith(AndroidJUnit4::class)
class EmbeddingGemmaEmbedderTest {

    companion object {
        private const val TAG = "EmbeddingGemmaEmbedderTest"

        private val REALISTIC_CLINICAL_CHUNK = """
            HISTORY OF PRESENT ILLNESS: This is a 64-year-old male with a history of
            hypertension, hyperlipidemia, and type 2 diabetes mellitus who presented to
            [**Hospital1 18**] on [**2151-3-14**] with complaints of substernal chest
            pressure radiating to the left arm, associated with diaphoresis and shortness
            of breath. Symptoms began approximately 2 hours prior to arrival while at rest.
            He denies nausea, vomiting, or lightheadedness. Of note, the patient has a
            family history of coronary artery disease, with his father having suffered a
            myocardial infarction at age 58. On arrival, EKG showed ST elevations in leads
            II, III, and aVF, consistent with an inferior STEMI. He was taken emergently to
            the cardiac catheterization lab, where a 100% occlusion of the right coronary
            artery was identified and successfully treated with a drug-eluting stent.
            Post-procedure, the patient was transferred to the CCU for monitoring. His
            medications on admission included Lisinopril 10mg daily, Atorvastatin 40mg
            daily, and Metformin 500mg twice daily. Follow up with Dr. [**Last Name
            (STitle) 4521**] was arranged for [**2151-3-28**].
        """.trimIndent()

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
    fun testMediaPipeEmbedderProducesSaneEmbeddings() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val embedder = EmbeddingGemmaEmbedder(context)

        try {
            val query = embedder.generateEmbedding(
                "What was the result of the cardiac catheterization?", isQuery = true
            )
            val paraphraseDoc = embedder.generateEmbedding(
                "Cardiac catheterization revealed a 100% occlusion of the right coronary artery.",
                isQuery = false
            )
            val unrelatedDoc = embedder.generateEmbedding(
                "The patient enjoys gardening and reading on weekends.", isQuery = false
            )

            Log.i(TAG, "query: ${query.embeddingFingerprint()}")
            Log.i(TAG, "paraphraseDoc: ${paraphraseDoc.embeddingFingerprint()}")
            Log.i(TAG, "unrelatedDoc: ${unrelatedDoc.embeddingFingerprint()}")

            assertEquals(768, query.size)
            assertEquals(768, paraphraseDoc.size)
            assertFalse("Embedding must not contain NaN", query.any { it.isNaN() })
            assertFalse("Embedding must not contain Inf", query.any { it.isInfinite() })

            val simRelated = cosineSimilarity(query, paraphraseDoc)
            val simUnrelated = cosineSimilarity(query, unrelatedDoc)
            Log.i(TAG, "cosine(query, paraphraseDoc)=$simRelated cosine(query, unrelatedDoc)=$simUnrelated")

            assertTrue(
                "Expected the paraphrased/related document to score higher than the unrelated one " +
                    "(related=$simRelated, unrelated=$simUnrelated)",
                simRelated > simUnrelated
            )
        } finally {
            embedder.close()
        }
    }

    @Test
    fun testMediaPipeEmbedderHandlesRealisticClinicalChunk() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val mpEmbedder = EmbeddingGemmaEmbedder(context)

        try {
            val newVec = mpEmbedder.generateEmbedding(REALISTIC_CLINICAL_CHUNK, isQuery = false)

            Log.i(TAG, "embedding: ${newVec.embeddingFingerprint()}")

            assertEquals(768, newVec.size)
            assertFalse(newVec.any { it.isNaN() })
        } finally {
            mpEmbedder.close()
        }
    }
}
