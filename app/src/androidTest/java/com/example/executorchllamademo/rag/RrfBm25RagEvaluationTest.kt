package com.example.executorchllamademo.rag

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.rag.data.MyObjectBox
import com.example.executorchllamademo.rag.data.ScoredChunk
import com.example.executorchllamademo.rag.embedding.Embedder
import com.example.executorchllamademo.rag.embedding.EmbeddingGemmaEmbedder
import com.example.executorchllamademo.rag.repository.HybridBm25MediaPipeRepository
import io.objectbox.BoxStore
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule
import java.io.Closeable
import java.io.File
import kotlin.math.sqrt

/** Retrieval AND generation evaluation for [RrfBm25MediaPipeRagEngine] - the paper's §4.3 Eq. 1 Hybrid row (BM25 + EmbeddingGemmaEmbedder fused by reciprocal rank fusion) - using the same dataset/pipeline/scoring as [MedicalRagEvaluationTest]. */
@RunWith(AndroidJUnit4::class)
class RrfBm25RagEvaluationTest {

    companion object {
        private const val TAG = "RrfBm25RagEvalTest"
        private const val ASSET_DIR = "rag_benchmark"
        private const val DEFAULT_DATASET_ASSET = "QA/single_dataset_rulebased.jsonl"
        private const val DEFAULT_PDF_SUBDIR = "Ammissioni_PDF_stratified_enriched"
        private const val TEST_DB_NAME = "objectbox-rrf-bm25-rag-evaluation-test"

        private const val RETRIEVAL_TOP_K = 10
        private const val GENERATION_TOP_K = 3
        private const val CONTEXT_HIT_THRESHOLD = 0.5
        private const val SEMANTIC_PASS_THRESHOLD = 0.75

        private const val READER_RESOURCE_PATH = "/data/local/tmp/llama/"
        private const val DEFAULT_READER_MODEL_FILE = "model.pte"
        private const val DEFAULT_READER_TOKENIZER_FILE = "tokenizer.json"
        private const val READER_MAX_SEQ_LEN = 768
        private const val READER_TEMPERATURE = 0.0f
        private const val CHATML_STOP_TOKEN = "<|im_end|>"

        private val ADMISSION_ID_REGEX = Regex("Admission_(\\d+)")
        private val TOKEN_REGEX = Regex("[a-zA-Z0-9]+")

        private fun distinctiveTokens(text: String): Set<String> =
            TOKEN_REGEX.findAll(text.lowercase()).map { it.value }.filter { it.length > 2 }.toSet()

        private fun overlapCoefficient(a: Set<String>, b: Set<String>): Double {
            if (a.isEmpty() || b.isEmpty()) return 0.0
            val intersection = a.count { it in b }
            return intersection.toDouble() / minOf(a.size, b.size)
        }

        private fun cosineSimilarity(a: FloatArray, b: FloatArray): Double {
            var dot = 0.0
            var normA = 0.0
            var normB = 0.0
            for (i in a.indices) {
                dot += a[i] * b[i]
                normA += a[i] * a[i]
                normB += b[i] * b[i]
            }
            if (normA == 0.0 || normB == 0.0) return 0.0
            return dot / (sqrt(normA) * sqrt(normB))
        }
    }

    data class BenchmarkItem(
        val id: String,
        val patientId: String,
        val documentName: String,
        val groundTruthContext: String,
        val question: String,
        val groundTruthAnswer: String,
        val section: String,
        val questionType: String
    )

    data class ItemResult(
        val id: String,
        val section: String,
        val questionType: String,
        val hitAt1: Boolean,
        val hitAt3: Boolean,
        val hitAt5: Boolean,
        val precisionAt3: Double,
        val precisionAt5: Double,
        val precisionAt10: Double,
        val ndcgAt3: Double,
        val ndcgAt10: Double,
        val reciprocalRank: Double,
        val averagePrecision: Double,
        val topChunkPreview: String,
        val generatedAnswer: String?,
        val semanticSimilarity: Double?,
        val generationPass: Boolean?
    )

    private lateinit var boxStore: BoxStore
    private lateinit var vectorRepository: HybridBm25MediaPipeRepository
    private lateinit var embedder: Embedder
    private lateinit var pdfProcessor: ClinicalPdfProcessor
    private lateinit var ragEngine: RrfBm25MediaPipeRagEngine
    private var readerModule: LlmModule? = null

    @Before
    fun setUp() {
        Log.i(TAG, "setUp: start")
        val targetContext = InstrumentationRegistry.getInstrumentation().targetContext

        ContextCompat.startForegroundService(
            targetContext,
            Intent(targetContext, BenchmarkKeepAliveService::class.java)
        )

        BoxStore.deleteAllFiles(targetContext.applicationContext, TEST_DB_NAME)
        boxStore = MyObjectBox.builder()
            .androidContext(targetContext.applicationContext)
            .name(TEST_DB_NAME)
            .build()
        val repo = HybridBm25MediaPipeRepository(boxStore)
        vectorRepository = repo
        Log.i(TAG, "setUp: constructing embedder (backend=mediapipe)")
        embedder = EmbeddingGemmaEmbedder(targetContext)
        Log.i(TAG, "setUp: embedder ready (backendId=${embedder.backendId})")
        Log.i(TAG, "setUp: constructing ClinicalPdfProcessor (PDFBox init)")
        pdfProcessor = ClinicalPdfProcessor(targetContext)

        ragEngine = RrfBm25MediaPipeRagEngine(repo, embedder)

        val args = InstrumentationRegistry.getArguments()
        val skipGeneration = (args.getString("skipGeneration", "false") ?: "false").toBoolean()
        readerModule = if (skipGeneration) {
            Log.i(TAG, "skipGeneration=true - running retrieval-only evaluation")
            null
        } else {
            Log.i(TAG, "setUp: loading reader LlmModule")
            loadReaderModule(args)
        }
        Log.i(TAG, "setUp: done")
    }

    @After
    fun tearDown() {
        (embedder as? Closeable)?.close()
        boxStore.close()
        val targetContext = InstrumentationRegistry.getInstrumentation().targetContext
        BoxStore.deleteAllFiles(targetContext.applicationContext, TEST_DB_NAME)
        targetContext.stopService(Intent(targetContext, BenchmarkKeepAliveService::class.java))
    }

    private fun loadReaderModule(args: Bundle): LlmModule? {
        val modelFile = args.getString("modelFile", DEFAULT_READER_MODEL_FILE)
            ?: DEFAULT_READER_MODEL_FILE
        val tokenizerFile = args.getString("tokenizerFile", DEFAULT_READER_TOKENIZER_FILE)
            ?: DEFAULT_READER_TOKENIZER_FILE
        val modelPath = READER_RESOURCE_PATH + modelFile
        val tokenizerPath = READER_RESOURCE_PATH + tokenizerFile

        if (!File(modelPath).exists() || !File(tokenizerPath).exists()) {
            Log.w(
                TAG,
                "Reader model not found at $modelPath / $tokenizerPath - skipping generation " +
                    "evaluation (retrieval metrics are unaffected)."
            )
            return null
        }

        return try {
            LlmModule(modelPath, tokenizerPath, READER_TEMPERATURE).apply { load() }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load reader model, skipping generation evaluation: ${e.message}")
            null
        }
    }

    private fun buildFactualQaPrompt(question: String, contexts: List<String>): String {
        val contextBlock = if (contexts.isEmpty()) {
            "(no relevant notes were retrieved)"
        } else {
            contexts.joinToString("\n\n") { "- $it" }
        }
        val systemPrompt = "You are a factual question-answering assistant for clinical " +
            "records. Answer the question using ONLY the clinical notes provided below. Be " +
            "concise and answer directly - do not add disclaimers, warnings, or commentary. " +
            "If the notes do not contain the answer, say so briefly.\n\n/no_think"
        return """
<|im_start|>system
$systemPrompt<|im_end|>
<|im_start|>user
CLINICAL NOTES:
$contextBlock

QUESTION: $question<|im_end|>
<|im_start|>assistant
""".trimStart()
    }

    private fun generateAnswer(module: LlmModule, prompt: String): String {
        val builder = StringBuilder()
        val callback = object : LlmCallback {
            override fun onResult(result: String) {
                if (result == CHATML_STOP_TOKEN) {
                    module.stop()
                    return
                }
                builder.append(result)
            }

            override fun onStats(result: String) {
            }
        }
        module.generate(prompt, READER_MAX_SEQ_LEN, callback, false)
        return builder.toString().trim()
    }

    private fun parseItem(line: String): BenchmarkItem {
        val obj = JSONObject(line)
        return BenchmarkItem(
            id = obj.getString("id"),
            patientId = obj.getString("patient_id"),
            documentName = obj.getString("document_name"),
            groundTruthContext = obj.getString("ground_truth_context"),
            question = obj.getString("question"),
            groundTruthAnswer = obj.getString("ground_truth_answer"),
            section = obj.getString("section"),
            questionType = obj.getString("question_type")
        )
    }

    private fun copyAssetToCache(assetPath: String, targetContext: Context): File {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val outFile = File(targetContext.cacheDir, assetPath.substringAfterLast('/'))
        testContext.assets.open(assetPath).use { input ->
            outFile.outputStream().use { output -> input.copyTo(output) }
        }
        return outFile
    }

    @Test
    fun runRrfBm25RagEvaluation() {
        val targetContext = InstrumentationRegistry.getInstrumentation().targetContext
        val testContext = InstrumentationRegistry.getInstrumentation().context

        val args = InstrumentationRegistry.getArguments()
        val maxItems = args.getString("maxItems", "")?.toIntOrNull()
        val startIndex = args.getString("startIndex", "")?.toIntOrNull() ?: 0
        val chunkSize = args.getString("chunkSize", "")?.toIntOrNull() ?: 500
        val overlapChars = args.getString("overlapChars", "")?.toIntOrNull() ?: 100
        val datasetAsset = "$ASSET_DIR/" + (args.getString("datasetAsset", "") ?: "").ifBlank { DEFAULT_DATASET_ASSET }
        val pdfAssetDir = "$ASSET_DIR/" + (args.getString("pdfAssetDir", "") ?: "").ifBlank { DEFAULT_PDF_SUBDIR }

        val items = testContext.assets.open(datasetAsset).bufferedReader().use { it.readLines() }
            .filter { it.isNotBlank() }
            .map { parseItem(it) }
            .drop(startIndex)
            .let { if (maxItems != null) it.take(maxItems) else it }
        assertTrue("Dataset asset was empty", items.isNotEmpty())
        Log.i(
            TAG,
            "Loaded ${items.size} benchmark items" +
                (if (startIndex > 0) " (starting at index $startIndex)" else "") +
                (maxItems?.let { " (capped at $it via maxItems)" } ?: "") +
                " (chunkSize=$chunkSize, overlapChars=$overlapChars)"
        )

        for ((index, item) in items.withIndex()) {
            val pdfFile = copyAssetToCache("$pdfAssetDir/${item.documentName}", targetContext)
            val rawText = runBlocking { pdfProcessor.extractText(Uri.fromFile(pdfFile)) }
            val chunks = pdfProcessor.createClinicalChunks(rawText, maxChars = chunkSize, overlapChars = overlapChars)
            val admissionId = ADMISSION_ID_REGEX.find(item.documentName)?.groupValues?.get(1)
                ?: item.documentName

            val clinicalChunks = chunks.map { chunkText ->
                ClinicalChunk(
                    patientId = item.patientId,
                    documentType = "Clinical Discharge Summary",
                    documentId = admissionId,
                    content = chunkText
                ).apply { embedding = embedder.generateEmbedding(chunkText) }
            }
            vectorRepository.insertChunks(clinicalChunks)
            pdfFile.delete()
            Log.i(TAG, "[${index + 1}/${items.size}] Ingested ${item.documentName}: ${chunks.size} chunks")
        }

        if (readerModule == null) {
            Log.i(TAG, "No reader model loaded - generation metrics will be omitted")
        }

        val results = mutableListOf<ItemResult>()
        for ((index, item) in items.withIndex()) {
            val gtTokens = distinctiveTokens(item.groundTruthContext)

            val ragOutput = runBlocking {
                ragEngine.preparePrompt(
                    userQuery = item.question,
                    patientId = item.patientId,
                    topK = RETRIEVAL_TOP_K,
                    maxContextCharsPerChunk = 2000
                )
            }

            val retrieved: List<ScoredChunk> = ragOutput.retrievedChunks
            val relevance: List<Boolean> = retrieved.map {
                overlapCoefficient(distinctiveTokens(it.chunk.content), gtTokens) >= CONTEXT_HIT_THRESHOLD
            }

            var generatedAnswer: String? = null
            var semanticSimilarity: Double? = null
            var generationPass: Boolean? = null
            val module = readerModule
            if (module != null) {
                module.resetContext()
                val genContexts = retrieved.take(GENERATION_TOP_K).map { it.chunk.content }
                val evalPrompt = buildFactualQaPrompt(item.question, genContexts)
                generatedAnswer = generateAnswer(module, evalPrompt)
                if (generatedAnswer.isBlank()) {
                    semanticSimilarity = 0.0
                    generationPass = false
                } else {
                    val answerVector = embedder.generateEmbedding(generatedAnswer)
                    val gtAnswerVector = embedder.generateEmbedding(item.groundTruthAnswer)
                    semanticSimilarity = cosineSimilarity(answerVector, gtAnswerVector)
                    generationPass = semanticSimilarity >= SEMANTIC_PASS_THRESHOLD
                }
            }

            val result = ItemResult(
                id = item.id,
                section = item.section,
                questionType = item.questionType,
                hitAt1 = RetrievalMetrics.hitAtK(relevance, 1),
                hitAt3 = RetrievalMetrics.hitAtK(relevance, 3),
                hitAt5 = RetrievalMetrics.hitAtK(relevance, 5),
                precisionAt3 = RetrievalMetrics.precisionAtK(relevance, 3),
                precisionAt5 = RetrievalMetrics.precisionAtK(relevance, 5),
                precisionAt10 = RetrievalMetrics.precisionAtK(relevance, 10),
                ndcgAt3 = RetrievalMetrics.ndcgAt(relevance, 3),
                ndcgAt10 = RetrievalMetrics.ndcgAt(relevance, 10),
                reciprocalRank = RetrievalMetrics.reciprocalRank(relevance),
                averagePrecision = RetrievalMetrics.averagePrecision(relevance),
                topChunkPreview = retrieved.firstOrNull()?.chunk?.content?.take(120) ?: "",
                generatedAnswer = generatedAnswer,
                semanticSimilarity = semanticSimilarity,
                generationPass = generationPass
            )
            results.add(result)
            Log.i(
                TAG,
                "[${index + 1}/${items.size}] [${result.id}] hit@1=${result.hitAt1} hit@3=${result.hitAt3} hit@5=${result.hitAt5} " +
                    "p@3=%.2f ndcg@3=%.2f ndcg@10=%.2f rr=%.2f semSim=%s".format(
                        result.precisionAt3, result.ndcgAt3, result.ndcgAt10, result.reciprocalRank,
                        result.semanticSimilarity?.let { "%.2f".format(it) } ?: "n/a"
                    )
            )
        }

        val n = results.size
        val hitRateAt1 = results.count { it.hitAt1 }.toDouble() / n
        val hitRateAt3 = results.count { it.hitAt3 }.toDouble() / n
        val hitRateAt5 = results.count { it.hitAt5 }.toDouble() / n
        val meanPrecisionAt3 = results.sumOf { it.precisionAt3 } / n
        val meanPrecisionAt5 = results.sumOf { it.precisionAt5 } / n
        val meanPrecisionAt10 = results.sumOf { it.precisionAt10 } / n
        val meanNdcgAt3 = results.sumOf { it.ndcgAt3 } / n
        val meanNdcgAt10 = results.sumOf { it.ndcgAt10 } / n
        val mrr = results.sumOf { it.reciprocalRank } / n
        val map = results.sumOf { it.averagePrecision } / n

        val scoredGeneration = results.filter { it.semanticSimilarity != null }
        val avgSemanticSimilarity = if (scoredGeneration.isNotEmpty()) {
            scoredGeneration.sumOf { it.semanticSimilarity!! } / scoredGeneration.size
        } else null
        val generationPassRate = if (scoredGeneration.isNotEmpty()) {
            scoredGeneration.count { it.generationPass == true }.toDouble() / scoredGeneration.size
        } else null

        Log.i(
            TAG,
            "SUMMARY: n=$n hit_rate@1=%.1f%% hit_rate@3=%.1f%% hit_rate@5=%.1f%% p@3=%.3f p@5=%.3f p@10=%.3f ndcg@3=%.3f ndcg@10=%.3f mrr=%.3f map=%.3f avg_sem_sim=%s pass_rate=%s".format(
                hitRateAt1 * 100, hitRateAt3 * 100, hitRateAt5 * 100, meanPrecisionAt3, meanPrecisionAt5, meanPrecisionAt10, meanNdcgAt3, meanNdcgAt10, mrr, map,
                avgSemanticSimilarity?.let { "%.1f%%".format(it * 100) } ?: "n/a (generation skipped)",
                generationPassRate?.let { "%.1f%%".format(it * 100) } ?: "n/a (generation skipped)"
            )
        )

        writeResultsJson(datasetAsset, results, n, hitRateAt1, hitRateAt3, hitRateAt5, meanPrecisionAt3, meanPrecisionAt5, meanPrecisionAt10, meanNdcgAt3, meanNdcgAt10, mrr, map, avgSemanticSimilarity, generationPassRate)
        assertTrue("No items were scored", results.isNotEmpty())
    }

    private fun writeResultsJson(
        datasetAsset: String,
        results: List<ItemResult>,
        n: Int,
        hitRateAt1: Double,
        hitRateAt3: Double,
        hitRateAt5: Double,
        meanPrecisionAt3: Double,
        meanPrecisionAt5: Double,
        meanPrecisionAt10: Double,
        meanNdcgAt3: Double,
        meanNdcgAt10: Double,
        mrr: Double,
        map: Double,
        avgSemanticSimilarity: Double?,
        generationPassRate: Double?
    ) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val datasetStem = datasetAsset.substringAfterLast('/').substringBeforeLast('.')
        val outFileName = "rrf_bm25_rag_results_$datasetStem.json"
        val additionalOutputDir = InstrumentationRegistry.getArguments().getString("additionalTestOutputDir")
        val outFile = if (additionalOutputDir != null) {
            File(additionalOutputDir, outFileName)
        } else {
            File(context.getExternalFilesDir(null), outFileName)
        }

        val root = JSONObject()
        root.put("num_items", n)
        root.put("hit_rate_at_1", hitRateAt1)
        root.put("hit_rate_at_3", hitRateAt3)
        root.put("hit_rate_at_5", hitRateAt5)
        root.put("precision_at_3", meanPrecisionAt3)
        root.put("precision_at_5", meanPrecisionAt5)
        root.put("precision_at_10", meanPrecisionAt10)
        root.put("ndcg_at_3", meanNdcgAt3)
        root.put("ndcg_at_10", meanNdcgAt10)
        root.put("mrr", mrr)
        root.put("map", map)
        root.put("generation_evaluated", avgSemanticSimilarity != null)
        root.put("avg_semantic_similarity", avgSemanticSimilarity ?: JSONObject.NULL)
        root.put("generation_pass_rate", generationPassRate ?: JSONObject.NULL)

        val itemsJson = JSONArray()
        for (r in results) {
            itemsJson.put(
                JSONObject().apply {
                    put("id", r.id)
                    put("section", r.section)
                    put("question_type", r.questionType)
                    put("hit_at_1", r.hitAt1)
                    put("hit_at_3", r.hitAt3)
                    put("hit_at_5", r.hitAt5)
                    put("precision_at_3", r.precisionAt3)
                    put("precision_at_5", r.precisionAt5)
                    put("precision_at_10", r.precisionAt10)
                    put("ndcg_at_3", r.ndcgAt3)
                    put("ndcg_at_10", r.ndcgAt10)
                    put("reciprocal_rank", r.reciprocalRank)
                    put("average_precision", r.averagePrecision)
                    put("top_chunk_preview", r.topChunkPreview)
                    put("generated_answer", r.generatedAnswer ?: JSONObject.NULL)
                    put("semantic_similarity", r.semanticSimilarity ?: JSONObject.NULL)
                    put("generation_pass", r.generationPass ?: JSONObject.NULL)
                }
            )
        }
        root.put("items", itemsJson)

        outFile.writeText(root.toString(2))
        Log.i(TAG, "Results written to ${outFile.path}")
    }
}
