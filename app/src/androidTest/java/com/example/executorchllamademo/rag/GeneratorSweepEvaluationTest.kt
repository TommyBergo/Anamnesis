package com.example.executorchllamademo.rag

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.executorchllamademo.BackendType
import com.example.executorchllamademo.ModelType
import com.example.executorchllamademo.ModelUtils
import com.example.executorchllamademo.rag.embedding.EmbeddingGemmaEmbedder
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
import java.io.File
import kotlin.math.sqrt

/** Three non-retrieval generator-sweep conditions (Closed-Book, Full-Context, Oracle-Evidence) that fill out the paper's Table 6 alongside the RAG evaluation tests. */
@RunWith(AndroidJUnit4::class)
class GeneratorSweepEvaluationTest {

    companion object {
        private const val TAG = "GeneratorSweepEvalTest"
        private const val ASSET_DIR = "rag_benchmark"
        private const val DEFAULT_DATASET_ASSET = "QA/single_dataset_llm.jsonl"
        private const val DEFAULT_PDF_SUBDIR = "Ammissioni_PDF_stratified_enriched"

        private const val SEMANTIC_PASS_THRESHOLD = 0.75

        private const val READER_RESOURCE_PATH = "/data/local/tmp/llama/"
        private const val DEFAULT_READER_MODEL_FILE = "model.pte"
        private const val DEFAULT_READER_TOKENIZER_FILE = "tokenizer.json"
        private const val CLOSED_BOOK_MAX_SEQ_LEN = 768
        private const val ORACLE_MAX_SEQ_LEN = 2048
        private const val FULL_CONTEXT_MAX_SEQ_LEN = 2048
        private const val FULL_CONTEXT_CHAR_BUDGET = 3000
        private const val READER_TEMPERATURE = 0.0f
        private const val CHATML_STOP_TOKEN = "<|im_end|>"
        private const val LLAMA3_STOP_TOKEN = "<|eot_id|>"
        private const val PHI4_STOP_TOKEN = "<|end|>"
        private const val GEMMA3_STOP_TOKEN = "<end_of_turn>"
        private val VALID_PROMPT_TEMPLATES = setOf("chatml", "llama3", "phi4", "gemma3")

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
        val generatedAnswer: String?,
        val semanticSimilarity: Double?,
        val generationPass: Boolean?,
        val error: String? = null
    )

    private var embedder: EmbeddingGemmaEmbedder? = null
    private lateinit var pdfProcessor: ClinicalPdfProcessor
    private var readerModule: LlmModule? = null

    private var promptTemplate: String = "chatml"

    private var oracleMaxSeqLen: Int = ORACLE_MAX_SEQ_LEN
    private var fullContextMaxSeqLen: Int = FULL_CONTEXT_MAX_SEQ_LEN
    private var fullContextCharBudget: Int = FULL_CONTEXT_CHAR_BUDGET

    @Before
    fun setUp() {
        Log.i(TAG, "setUp: start")
        val targetContext = InstrumentationRegistry.getInstrumentation().targetContext

        ContextCompat.startForegroundService(
            targetContext,
            Intent(targetContext, BenchmarkKeepAliveService::class.java)
        )

        pdfProcessor = ClinicalPdfProcessor(targetContext)

        val args = InstrumentationRegistry.getArguments()
        val scoreAnswers = (args.getString("scoreAnswers", "false") ?: "false").toBoolean()
        embedder = if (scoreAnswers) {
            EmbeddingGemmaEmbedder(targetContext)
        } else {
            Log.i(TAG, "setUp: scoreAnswers=false - skipping embedder, results carry generated_answer only")
            null
        }
        promptTemplate = args.getString("promptTemplate", "chatml") ?: "chatml"
        require(promptTemplate in VALID_PROMPT_TEMPLATES) {
            "Unknown promptTemplate '$promptTemplate' - expected one of $VALID_PROMPT_TEMPLATES"
        }
        Log.i(TAG, "setUp: promptTemplate=$promptTemplate")
        oracleMaxSeqLen = args.getString("oracleMaxSeqLen", "")?.toIntOrNull() ?: ORACLE_MAX_SEQ_LEN
        fullContextMaxSeqLen = args.getString("fullContextMaxSeqLen", "")?.toIntOrNull() ?: FULL_CONTEXT_MAX_SEQ_LEN
        fullContextCharBudget = args.getString("fullContextCharBudget", "")?.toIntOrNull() ?: FULL_CONTEXT_CHAR_BUDGET
        Log.i(TAG, "setUp: oracleMaxSeqLen=$oracleMaxSeqLen fullContextMaxSeqLen=$fullContextMaxSeqLen fullContextCharBudget=$fullContextCharBudget")
        readerModule = loadReaderModule(args)
        Log.i(TAG, "setUp: done")
    }

    @After
    fun tearDown() {
        embedder?.close()
        val targetContext = InstrumentationRegistry.getInstrumentation().targetContext
        targetContext.stopService(Intent(targetContext, BenchmarkKeepAliveService::class.java))
    }

    private fun loadReaderModule(args: Bundle): LlmModule? {
        val modelFile = args.getString("modelFile", DEFAULT_READER_MODEL_FILE) ?: DEFAULT_READER_MODEL_FILE
        val tokenizerFile = args.getString("tokenizerFile", DEFAULT_READER_TOKENIZER_FILE) ?: DEFAULT_READER_TOKENIZER_FILE
        val dataFilesArg = args.getString("dataFiles", "") ?: ""
        val dataFiles = dataFilesArg.split(",").map { it.trim() }.filter { it.isNotBlank() }
        val modelPath = READER_RESOURCE_PATH + modelFile
        val tokenizerPath = READER_RESOURCE_PATH + tokenizerFile
        val dataPaths = dataFiles.map { READER_RESOURCE_PATH + it }

        if (!File(modelPath).exists() || !File(tokenizerPath).exists() ||
            dataPaths.any { !File(it).exists() }
        ) {
            Log.w(TAG, "Reader model not found at $modelPath / $tokenizerPath - cannot run generator sweep.")
            return null
        }

        return try {
            val module = if (dataPaths.isNotEmpty()) {
                LlmModule(
                    ModelUtils.getModelCategory(ModelType.LLAMA_3, BackendType.XNNPACK),
                    modelPath,
                    tokenizerPath,
                    READER_TEMPERATURE,
                    dataPaths
                )
            } else {
                LlmModule(modelPath, tokenizerPath, READER_TEMPERATURE)
            }
            module.apply { load() }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load reader model: ${e.message}")
            null
        }
    }

    private fun wrapPrompt(systemPrompt: String, userBody: String): String {
        return when (promptTemplate) {
            "llama3" ->
                """
<|start_header_id|>system<|end_header_id|>

$systemPrompt<|eot_id|><|start_header_id|>user<|end_header_id|>

$userBody<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""".trimStart()
            "phi4" ->
                """
<|system|>$systemPrompt<|end|><|user|>$userBody<|end|><|assistant|>""".trimStart()
            "gemma3" ->
                """
<start_of_turn>user
$systemPrompt

$userBody<end_of_turn>
<start_of_turn>model
""".trimStart()
            else ->
                """
<|im_start|>system
$systemPrompt

/no_think<|im_end|>
<|im_start|>user
$userBody<|im_end|>
<|im_start|>assistant
""".trimStart()
        }
    }

    private fun buildClosedBookPrompt(question: String): String {
        val systemPrompt = "You are a factual medical question-answering assistant. Answer the " +
            "question using your own medical knowledge - no clinical notes are provided for this " +
            "question. Be concise and answer directly - do not add disclaimers, warnings, or " +
            "commentary. If you cannot answer confidently, say so briefly."
        return wrapPrompt(systemPrompt, "QUESTION: $question")
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
            "If the notes do not contain the answer, say so briefly."
        return wrapPrompt(systemPrompt, "CLINICAL NOTES:\n$contextBlock\n\nQUESTION: $question")
    }

    private fun generateAnswer(module: LlmModule, prompt: String, maxSeqLen: Int): Result<String> {
        val stopToken = when (promptTemplate) {
            "llama3" -> LLAMA3_STOP_TOKEN
            "phi4" -> PHI4_STOP_TOKEN
            "gemma3" -> GEMMA3_STOP_TOKEN
            else -> CHATML_STOP_TOKEN
        }
        val builder = StringBuilder()
        val callback = object : LlmCallback {
            override fun onResult(result: String) {
                if (result == stopToken) {
                    module.stop()
                    return
                }
                builder.append(result)
            }

            override fun onStats(result: String) {}
        }
        return try {
            module.generate(prompt, maxSeqLen, callback, false)
            Result.success(builder.toString().trim())
        } catch (e: Exception) {
            Result.failure(e)
        }
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

    private fun loadItems(): List<BenchmarkItem> {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val args = InstrumentationRegistry.getArguments()
        val maxItems = args.getString("maxItems", "")?.toIntOrNull()
        val startIndex = args.getString("startIndex", "")?.toIntOrNull() ?: 0
        val datasetAsset = "$ASSET_DIR/" + (args.getString("datasetAsset", "") ?: "").ifBlank { DEFAULT_DATASET_ASSET }

        val items = testContext.assets.open(datasetAsset).bufferedReader().use { it.readLines() }
            .filter { it.isNotBlank() }
            .map { parseItem(it) }
            .drop(startIndex)
            .let { if (maxItems != null) it.take(maxItems) else it }
        assertTrue("Dataset asset was empty", items.isNotEmpty())
        Log.i(
            TAG,
            "Loaded ${items.size} benchmark items from $datasetAsset" +
                (if (startIndex > 0) " (starting at index $startIndex)" else "") +
                (maxItems?.let { " (capped at $it via maxItems)" } ?: "")
        )
        return items
    }

    private fun scoreAnswer(generatedAnswer: String, groundTruthAnswer: String): Pair<Double, Boolean> {
        val emb = requireNotNull(embedder) { "scoreAnswer called with no embedder - caller must guard on embedder == null" }
        val answerVector = emb.generateEmbedding(generatedAnswer)
        val gtAnswerVector = emb.generateEmbedding(groundTruthAnswer)
        val sim = cosineSimilarity(answerVector, gtAnswerVector)
        return sim to (sim >= SEMANTIC_PASS_THRESHOLD)
    }

    private fun runCondition(outFileName: String, contextFor: (BenchmarkItem) -> RunOutcome) {
        val module = readerModule
        assertTrue("Reader model not available - cannot run generator sweep", module != null)
        module!!

        val items = loadItems()
        val results = mutableListOf<ItemResult>()
        for ((index, item) in items.withIndex()) {
            module.resetContext()
            val outcome = contextFor(item)
            val result = when (outcome) {
                is RunOutcome.Skip -> ItemResult(
                    id = item.id, section = item.section, questionType = item.questionType,
                    generatedAnswer = null, semanticSimilarity = null, generationPass = null,
                    error = outcome.reason
                )
                is RunOutcome.Prompt -> {
                    val genResult = generateAnswer(module, outcome.prompt, outcome.maxSeqLen)
                    genResult.fold(
                        onSuccess = { answer ->
                            if (embedder == null) {
                                ItemResult(item.id, item.section, item.questionType, answer, null, null)
                            } else if (answer.isBlank()) {
                                ItemResult(item.id, item.section, item.questionType, answer, 0.0, false)
                            } else {
                                val (sim, pass) = scoreAnswer(answer, item.groundTruthAnswer)
                                ItemResult(item.id, item.section, item.questionType, answer, sim, pass)
                            }
                        },
                        onFailure = { e ->
                            ItemResult(
                                item.id, item.section, item.questionType, null, null, null,
                                error = "generation threw: ${e.message}"
                            )
                        }
                    )
                }
            }
            results.add(result)
            Log.i(
                TAG,
                "[${index + 1}/${items.size}] [${result.id}] semSim=${result.semanticSimilarity?.let { "%.2f".format(it) } ?: "n/a"} " +
                    "pass=${result.generationPass} error=${result.error}"
            )
        }

        val scored = results.filter { it.semanticSimilarity != null }
        val avgSemanticSimilarity = if (scored.isNotEmpty()) scored.sumOf { it.semanticSimilarity!! } / scored.size else null
        val generationPassRate = if (scored.isNotEmpty()) scored.count { it.generationPass == true }.toDouble() / scored.size else null
        val errorRate = results.count { it.error != null }.toDouble() / results.size

        Log.i(
            TAG,
            "SUMMARY[$outFileName]: n=${results.size} avg_sem_sim=%s pass_rate=%s error_rate=%.1f%%".format(
                avgSemanticSimilarity?.let { "%.1f%%".format(it * 100) } ?: "n/a",
                generationPassRate?.let { "%.1f%%".format(it * 100) } ?: "n/a",
                errorRate * 100
            )
        )

        writeResultsJson(outFileName, results, avgSemanticSimilarity, generationPassRate)
        assertTrue("No items were scored", results.isNotEmpty())
    }

    private sealed class RunOutcome {
        data class Prompt(val prompt: String, val maxSeqLen: Int) : RunOutcome()
        data class Skip(val reason: String) : RunOutcome()
    }

    @Test
    fun runClosedBookEvaluation() {
        runCondition("closed_book_results.json") { item ->
            RunOutcome.Prompt(buildClosedBookPrompt(item.question), CLOSED_BOOK_MAX_SEQ_LEN)
        }
    }

    @Test
    fun runOracleEvidenceEvaluation() {
        runCondition("oracle_evidence_results.json") { item ->
            RunOutcome.Prompt(
                buildFactualQaPrompt(item.question, listOf(item.groundTruthContext)),
                oracleMaxSeqLen
            )
        }
    }

    @Test
    fun runFullContextEvaluation() {
        val targetContext = InstrumentationRegistry.getInstrumentation().targetContext
        val args = InstrumentationRegistry.getArguments()
        val pdfAssetDir = "$ASSET_DIR/" + (args.getString("pdfAssetDir", "") ?: "").ifBlank { DEFAULT_PDF_SUBDIR }

        val docTextCache = mutableMapOf<String, String>()
        runCondition("full_context_results.json") { item ->
            val fullText = docTextCache.getOrPut(item.documentName) {
                try {
                    val pdfFile = copyAssetToCache("$pdfAssetDir/${item.documentName}", targetContext)
                    val text = runBlocking { pdfProcessor.extractText(Uri.fromFile(pdfFile)) }
                    pdfFile.delete()
                    text
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to extract full text for ${item.documentName}: ${e.message}")
                    ""
                }
            }
            if (fullText.isBlank()) {
                RunOutcome.Skip("could not extract text for ${item.documentName}")
            } else {
                val truncated = fullText.take(fullContextCharBudget)
                RunOutcome.Prompt(buildFactualQaPrompt(item.question, listOf(truncated)), fullContextMaxSeqLen)
            }
        }
    }

    private fun writeResultsJson(
        outFileName: String,
        results: List<ItemResult>,
        avgSemanticSimilarity: Double?,
        generationPassRate: Double?
    ) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val additionalOutputDir = InstrumentationRegistry.getArguments().getString("additionalTestOutputDir")
        val outFile = if (additionalOutputDir != null) {
            File(additionalOutputDir, outFileName)
        } else {
            File(context.getExternalFilesDir(null), outFileName)
        }

        val root = JSONObject()
        root.put("num_items", results.size)
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
                    put("generated_answer", r.generatedAnswer ?: JSONObject.NULL)
                    put("semantic_similarity", r.semanticSimilarity ?: JSONObject.NULL)
                    put("generation_pass", r.generationPass ?: JSONObject.NULL)
                    put("error", r.error ?: JSONObject.NULL)
                }
            )
        }
        root.put("items", itemsJson)

        outFile.writeText(root.toString(2))
        Log.i(TAG, "Results written to ${outFile.path}")
    }
}
