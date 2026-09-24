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
import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.rag.data.MyObjectBox
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
import java.io.File
import kotlin.math.sqrt

/** Multi-document evaluation of [RrfBm25MediaPipeRagEngine] (BM25 + EmbeddingGemmaEmbedder fused by reciprocal rank fusion, the paper's §4.3 Eq. 1 Hybrid row), the RRF counterpart to [MultiDocBm25RagEvaluationTest] and [MultiDocRagEvaluationTest]. */
@RunWith(AndroidJUnit4::class)
class RrfBm25MultiDocRagEvaluationTest {

    companion object {
        private const val TAG = "RrfBm25MultiDocRagEvalTest"
        private const val ASSET_DIR = "rag_benchmark"
        private const val DEFAULT_DATASET_ASSET = "QA/multidoc_dataset_llm_main_same_patient.jsonl"
        private const val DEFAULT_PDF_SUBDIR = "Ammissioni_PDF_stratified_enriched"
        private const val TEST_DB_NAME = "objectbox-rrf-bm25-multidoc-rag-evaluation-test"

        private const val RETRIEVAL_TOP_K = 10
        private const val WRONG_ADMISSION_TOP_K = 3
        private const val GENERATION_CANDIDATE_TOP_K_PER_PATIENT = RETRIEVAL_TOP_K
        private const val CONTEXT_HIT_THRESHOLD = 0.5
        private const val SEMANTIC_PASS_THRESHOLD = 0.75

        private const val READER_RESOURCE_PATH = "/data/local/tmp/llama/"
        private const val DEFAULT_READER_MODEL_FILE = "model.pte"
        private const val DEFAULT_READER_TOKENIZER_FILE = "tokenizer.json"
        private const val READER_MAX_SEQ_LEN = 2048
        private const val MIN_PARTIAL_CONTEXT_CHARS = 100
        private const val TRUNCATION_TOKEN_MARGIN = 8
        private const val READER_TEMPERATURE = 0.0f
        private const val CHATML_STOP_TOKEN = "<|im_end|>"

        private val TOKEN_REGEX = Regex("[a-zA-Z0-9]+")
        private val ADMISSION_ID_REGEX = Regex("Admission_(\\d+)")

        private fun distinctiveTokens(text: String): Set<String> =
            TOKEN_REGEX.findAll(text.lowercase())
                .map { it.value }
                .filter { it.length > 2 }
                .toSet()

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

    data class SourceRef(
        val patientId: String,
        val admissionId: String,
        val sourcePdf: String,
        val sourceChunk: String
    )

    data class MultiDocItem(
        val id: String,
        val comparisonType: String,
        val question: String,
        val referenceAnswer: String,
        val sources: List<SourceRef>
    )

    data class GenerationBudget(
        val answerReserve: Int,
        val charsPerToken: Double,
        val contextCharCap: Int
    )

    data class GenerationAttempt(
        val answer: String,
        val promptChars: Int,
        val contextBlocks: Int,
        val droppedBlocks: Int,
        val promptTokens: Int?,
        val generatedTokens: Int?,
        val likelyTruncated: Boolean,
        val sawChatmlStopToken: Boolean,
        val statsText: String?,
        val error: String?
    )

    data class GenerationDiagnostics(
        val promptChars: Int,
        val contextBlocks: Int,
        val droppedBlocks: Int,
        val retryUsed: Boolean,
        val attemptsUsed: Int,
        val promptTokens: Int?,
        val generatedTokens: Int?,
        val likelyTruncated: Boolean,
        val sawChatmlStopToken: Boolean,
        val statsText: String?,
        val error: String?
    )

    data class ItemResult(
        val id: String,
        val comparisonType: String,
        val numSources: Int,
        val numPatients: Int,
        val sourceHitAt1: Double,
        val sourceHitAt3: Double,
        val sourceHitAt5: Double,
        val sourceHitAt10: Double,
        val allSourcesHitAt1: Boolean,
        val allSourcesHitAt3: Boolean,
        val allSourcesHitAt5: Boolean,
        val allSourcesHitAt10: Boolean,
        val sourceMrrMean: Double,
        val sourceMrrWorst: Double,
        val sourcePrecisionAt3: Double,
        val sourcePrecisionAt5: Double,
        val sourcePrecisionAt10: Double,
        val sourceNdcgAt3: Double,
        val sourceNdcgAt10: Double,
        val sourceAveragePrecision: Double,
        val wrongAdmissionRateAt3: Double?,
        val wrongAdmissionRateAt10: Double?,
        val generatedAnswer: String?,
        val semanticSimilarity: Double?,
        val generationPass: Boolean?,
        val generationDiagnostics: GenerationDiagnostics?
    )

    private lateinit var boxStore: BoxStore
    private lateinit var vectorRepository: HybridBm25MediaPipeRepository
    private lateinit var embedder: EmbeddingGemmaEmbedder
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
        Log.i(TAG, "setUp: constructing EmbeddingGemmaEmbedder (retrieval dense side + scoring)")
        embedder = EmbeddingGemmaEmbedder(targetContext)
        pdfProcessor = ClinicalPdfProcessor(targetContext)
        ragEngine = RrfBm25MediaPipeRagEngine(repo, embedder)

        val args = InstrumentationRegistry.getArguments()
        val skipGeneration = (args.getString("skipGeneration", "false") ?: "false").toBoolean()
        readerModule = if (skipGeneration) {
            Log.i(TAG, "skipGeneration=true - running retrieval-only evaluation")
            null
        } else {
            loadReaderModule(args)
        }
        Log.i(TAG, "setUp: done")
    }

    @After
    fun tearDown() {
        embedder.close()
        boxStore.close()
        val targetContext = InstrumentationRegistry.getInstrumentation().targetContext
        BoxStore.deleteAllFiles(targetContext.applicationContext, TEST_DB_NAME)
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

        if (!File(modelPath).exists() || !File(tokenizerPath).exists() || dataPaths.any { !File(it).exists() }) {
            Log.w(
                TAG,
                "Reader model not found at $modelPath / $tokenizerPath - skipping generation evaluation " +
                    "(retrieval metrics are unaffected)."
            )
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
            Log.w(TAG, "Failed to load reader model, skipping generation evaluation: ${e.message}")
            null
        }
    }

    private fun parseItem(line: String): MultiDocItem {
        val obj = JSONObject(line)
        val sourcesJson = obj.getJSONArray("sources")
        val sources = (0 until sourcesJson.length()).map { i ->
            val s = sourcesJson.getJSONObject(i)
            SourceRef(
                patientId = s.getString("patient_id"),
                admissionId = s.getString("admission_id"),
                sourcePdf = s.getString("source_pdf"),
                sourceChunk = s.getString("source_chunk")
            )
        }
        return MultiDocItem(
            id = obj.getString("id"),
            comparisonType = obj.getString("comparison_type"),
            question = obj.getString("question"),
            referenceAnswer = obj.getString("reference_answer"),
            sources = sources
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

    private fun buildFactualQaPrompt(question: String, labelledContexts: List<String>): String {
        val contextBlock = if (labelledContexts.isEmpty()) {
            "(no relevant notes were retrieved)"
        } else {
            labelledContexts.joinToString("\n\n") { "- $it" }
        }
        val systemPrompt = "You are a factual question-answering assistant for clinical " +
            "records. Some questions compare facts across more than one admission or patient - " +
            "each note below is labelled with which patient/admission it comes from. Answer the " +
            "question using ONLY the clinical notes provided below. Be concise and answer " +
            "directly - do not add disclaimers, warnings, or commentary. If the notes do not " +
            "contain the answer, say so briefly.\n\n/no_think"
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

    private fun estimateTokens(text: String, charsPerToken: Double): Int =
        kotlin.math.ceil(text.length / charsPerToken).toInt()

    private fun packGenerationContexts(
        question: String,
        patientIds: List<String>,
        candidatesByPatient: Map<String, List<String>>,
        answerReserve: Int,
        charsPerToken: Double,
        contextCharCap: Int
    ): Pair<List<String>, Int> {
        val cappedByPatient = patientIds.associateWith { pid ->
            candidatesByPatient[pid].orEmpty()
                .take(GENERATION_CANDIDATE_TOP_K_PER_PATIENT)
                .map { block ->
                    val labelEnd = block.indexOf(": ") + 2
                    if (labelEnd >= 2) {
                        val label = block.substring(0, labelEnd)
                        val content = block.substring(labelEnd)
                        label + content.take(contextCharCap)
                    } else block.take(contextCharCap)
                }
        }
        val totalCandidates = cappedByPatient.values.sumOf { it.size }
        val ordered = mutableListOf<String>()
        for (rank in 0 until GENERATION_CANDIDATE_TOP_K_PER_PATIENT) {
            for (pid in patientIds) cappedByPatient[pid]?.getOrNull(rank)?.let { ordered.add(it) }
        }

        val promptTokenBudget = READER_MAX_SEQ_LEN - answerReserve
        val admitted = mutableListOf<String>()
        for (block in ordered) {
            if (estimateTokens(buildFactualQaPrompt(question, admitted + block), charsPerToken) <= promptTokenBudget) {
                admitted.add(block)
                continue
            }
            val currentPromptTokens = estimateTokens(buildFactualQaPrompt(question, admitted), charsPerToken)
            val remainingTokens = promptTokenBudget - currentPromptTokens
            if (remainingTokens <= 0) continue

            val labelEnd = block.indexOf(": ") + 2
            val label = if (labelEnd >= 2) block.substring(0, labelEnd) else ""
            val content = if (labelEnd >= 2) block.substring(labelEnd) else block
            val labelTokens = estimateTokens("\n\n- $label", charsPerToken)
            val charsAvailable = ((remainingTokens - labelTokens) * charsPerToken).toInt().coerceAtMost(content.length)
            if (charsAvailable >= MIN_PARTIAL_CONTEXT_CHARS) {
                val partial = label + content.take(charsAvailable)
                if (estimateTokens(buildFactualQaPrompt(question, admitted + partial), charsPerToken) <= promptTokenBudget) {
                    admitted.add(partial)
                }
            }
        }
        return admitted to (totalCandidates - admitted.size)
    }

    private fun parseStatInt(statsText: String?, key: String): Int? {
        if (statsText.isNullOrBlank()) return null
        try {
            val obj = JSONObject(statsText)
            if (obj.has(key) && !obj.isNull(key)) {
                val jsonValue = obj.optInt(key, -1)
                if (jsonValue >= 0) return jsonValue
            }
        } catch (_: Exception) {
        }
        val aliases = when (key) {
            "prompt_tokens" -> listOf("prompt_tokens", "prompt tokens")
            "generated_tokens" -> listOf("generated_tokens", "generated tokens")
            else -> listOf(key)
        }
        for (alias in aliases) {
            val regex = Regex("(?i)" + Regex.escape(alias) + "\\s*[:=]\\s*(\\d+)")
            val parsed = regex.find(statsText)?.groupValues?.getOrNull(1)?.toIntOrNull()
            if (parsed != null) return parsed
        }
        return null
    }

    private fun generateOnce(
        module: LlmModule,
        prompt: String,
        contextBlocks: Int,
        droppedBlocks: Int,
        itemId: String
    ): GenerationAttempt {
        val builder = StringBuilder()
        var sawStop = false
        var statsText: String? = null
        var error: String? = null
        val callback = object : LlmCallback {
            override fun onResult(result: String) {
                if (sawStop) return
                builder.append(result)
                val stopIndex = builder.indexOf(CHATML_STOP_TOKEN)
                if (stopIndex >= 0) {
                    sawStop = true
                    builder.setLength(stopIndex)
                    module.stop()
                }
            }
            override fun onStats(result: String) {
                statsText = result
                Log.i(TAG, "[$itemId] generationStats=$result")
            }
        }
        try {
            module.resetContext()
            module.generate(prompt, READER_MAX_SEQ_LEN, callback, false)
        } catch (e: Exception) {
            error = "${e.javaClass.simpleName}: ${e.message}"
            Log.e(TAG, "[$itemId] generation failed", e)
        }
        val promptTokens = parseStatInt(statsText, "prompt_tokens")
        val generatedTokens = parseStatInt(statsText, "generated_tokens")
        val likelyTruncated = promptTokens != null && generatedTokens != null &&
            promptTokens + generatedTokens >= READER_MAX_SEQ_LEN - TRUNCATION_TOKEN_MARGIN
        return GenerationAttempt(
            builder.toString().trim(), prompt.length, contextBlocks, droppedBlocks,
            promptTokens, generatedTokens, likelyTruncated, sawStop, statsText, error
        )
    }

    @Test
    fun runRrfBm25MultiDocRagEvaluation() {
        val targetContext = InstrumentationRegistry.getInstrumentation().targetContext
        val testContext = InstrumentationRegistry.getInstrumentation().context

        val maxItems = InstrumentationRegistry.getArguments().getString("maxItems", "")?.toIntOrNull()
        val startIndex = InstrumentationRegistry.getArguments().getString("startIndex", "")?.toIntOrNull() ?: 0
        val chunkSize = InstrumentationRegistry.getArguments().getString("chunkSize", "")?.toIntOrNull() ?: 500
        val overlapChars = InstrumentationRegistry.getArguments().getString("overlapChars", "")?.toIntOrNull() ?: 100
        val datasetAsset = "$ASSET_DIR/" +
            (InstrumentationRegistry.getArguments().getString("datasetAsset", "") ?: "").ifBlank { DEFAULT_DATASET_ASSET }
        val pdfAssetDir = "$ASSET_DIR/" +
            (InstrumentationRegistry.getArguments().getString("pdfAssetDir", "") ?: "").ifBlank { DEFAULT_PDF_SUBDIR }

        val items = testContext.assets.open(datasetAsset).bufferedReader().use { it.readLines() }
            .filter { it.isNotBlank() }
            .map { parseItem(it) }
            .drop(startIndex)
            .let { if (maxItems != null) it.take(maxItems) else it }
        assertTrue("Dataset asset was empty", items.isNotEmpty())
        Log.i(TAG, "Loaded ${items.size} multidoc benchmark items from $datasetAsset (chunkSize=$chunkSize, overlapChars=$overlapChars)")
        if (readerModule == null) {
            Log.i(TAG, "No reader model loaded - generation metrics will be omitted")
        }

        val distinctPatientIds = items.flatMap { it.sources }.map { it.patientId }.distinct()
        for ((index, patientId) in distinctPatientIds.withIndex()) {
            val patientPrefix = "Patient_${patientId}_"
            val patientDocs = (testContext.assets.list(pdfAssetDir) ?: emptyArray())
                .filter { it.startsWith(patientPrefix) }
            if (patientDocs.isEmpty()) {
                Log.w(TAG, "[${index + 1}/${distinctPatientIds.size}] No admission PDFs found for patient $patientId (prefix $patientPrefix)")
            }
            for (documentName in patientDocs) {
                val pdfFile = copyAssetToCache("$pdfAssetDir/$documentName", targetContext)
                val rawText = runBlocking { pdfProcessor.extractText(Uri.fromFile(pdfFile)) }
                val chunks = pdfProcessor.createClinicalChunks(rawText, maxChars = chunkSize, overlapChars = overlapChars)
                val admissionId = ADMISSION_ID_REGEX.find(documentName)?.groupValues?.get(1) ?: documentName
                val clinicalChunks = chunks.map { chunkText ->
                    ClinicalChunk(
                        patientId = patientId,
                        documentType = "Clinical Discharge Summary",
                        documentId = admissionId,
                        content = chunkText
                    ).apply { embedding = embedder.generateEmbedding(chunkText) }
                }
                vectorRepository.insertChunks(clinicalChunks)
                pdfFile.delete()
                Log.i(TAG, "[${index + 1}/${distinctPatientIds.size}] Ingested $documentName (patient $patientId): ${chunks.size} chunks")
            }
        }

        val results = mutableListOf<ItemResult>()
        for ((index, item) in items.withIndex()) {
            val patientIds = item.sources.map { it.patientId }.distinct()
            val perPatientRetrieved = patientIds.associateWith { pid ->
                runBlocking {
                    ragEngine.preparePrompt(
                        userQuery = item.question,
                        patientId = pid,
                        topK = RETRIEVAL_TOP_K,
                        maxContextCharsPerChunk = 2000
                    )
                }.retrievedChunks
            }

            var hitAt1Count = 0
            var hitAt3Count = 0
            var hitAt5Count = 0
            var hitAt10Count = 0
            val reciprocalRanks = mutableListOf<Double>()
            val precisionAt3Values = mutableListOf<Double>()
            val precisionAt5Values = mutableListOf<Double>()
            val precisionAt10Values = mutableListOf<Double>()
            val ndcgAt3Values = mutableListOf<Double>()
            val ndcgAt10Values = mutableListOf<Double>()
            val averagePrecisionValues = mutableListOf<Double>()
            for (source in item.sources) {
                val retrieved = perPatientRetrieved[source.patientId].orEmpty()
                val gtTokens = distinctiveTokens(source.sourceChunk)
                val relevance = retrieved.map {
                    overlapCoefficient(distinctiveTokens(it.chunk.content), gtTokens) >= CONTEXT_HIT_THRESHOLD
                }
                if (RetrievalMetrics.hitAtK(relevance, 1)) hitAt1Count++
                if (RetrievalMetrics.hitAtK(relevance, 3)) hitAt3Count++
                if (RetrievalMetrics.hitAtK(relevance, 5)) hitAt5Count++
                if (RetrievalMetrics.hitAtK(relevance, 10)) hitAt10Count++
                reciprocalRanks.add(RetrievalMetrics.reciprocalRank(relevance))

                precisionAt3Values.add(RetrievalMetrics.precisionAtK(relevance, 3))
                precisionAt5Values.add(RetrievalMetrics.precisionAtK(relevance, 5))
                precisionAt10Values.add(RetrievalMetrics.precisionAtK(relevance, 10))
                ndcgAt3Values.add(RetrievalMetrics.ndcgAt(relevance, 3))
                ndcgAt10Values.add(RetrievalMetrics.ndcgAt(relevance, 10))
                averagePrecisionValues.add(RetrievalMetrics.averagePrecision(relevance))
            }
            val numSources = item.sources.size
            val sourceHitAt1 = hitAt1Count.toDouble() / numSources
            val sourceHitAt3 = hitAt3Count.toDouble() / numSources
            val sourceHitAt5 = hitAt5Count.toDouble() / numSources
            val sourceHitAt10 = hitAt10Count.toDouble() / numSources
            val allSourcesHitAt1 = hitAt1Count == numSources
            val allSourcesHitAt3 = hitAt3Count == numSources
            val allSourcesHitAt5 = hitAt5Count == numSources
            val allSourcesHitAt10 = hitAt10Count == numSources
            val sourceMrrMean = reciprocalRanks.average()
            val sourceMrrWorst = reciprocalRanks.min()

            val requiredAdmissionsByPatient: Map<String, Set<String>> =
                item.sources.groupBy { it.patientId }.mapValues { (_, srcs) -> srcs.map { it.admissionId }.toSet() }
            fun wrongAdmissionRate(topK: Int): Double? {
                val samples = patientIds.mapNotNull { pid ->
                    val top = perPatientRetrieved[pid].orEmpty().take(topK)
                    val required = requiredAdmissionsByPatient[pid].orEmpty()
                    if (top.isEmpty()) null else top.count { it.chunk.documentId !in required }.toDouble() / top.size
                }
                return if (samples.isEmpty()) null else samples.average()
            }
            val wrongAdmissionRateAt3 = wrongAdmissionRate(WRONG_ADMISSION_TOP_K)
            val wrongAdmissionRateAt10 = wrongAdmissionRate(RETRIEVAL_TOP_K)

            var generatedAnswer: String? = null
            var semanticSimilarity: Double? = null
            var generationPass: Boolean? = null
            var generationDiagnostics: GenerationDiagnostics? = null
            val module = readerModule
            if (module != null) {
                val candidatesByPatient: Map<String, List<String>> = patientIds.associateWith { pid ->
                    perPatientRetrieved[pid].orEmpty()
                        .take(GENERATION_CANDIDATE_TOP_K_PER_PATIENT)
                        .map { scored ->
                            "[Patient $pid, admission ${scored.chunk.documentId}]: ${scored.chunk.content}"
                        }
                }
                val budgets = listOf(
                    GenerationBudget(384, 2.0, 500),
                    GenerationBudget(640, 1.8, 320),
                    GenerationBudget(896, 1.6, 220),
                    GenerationBudget(1152, 1.4, 160)
                )
                var finalAttempt: GenerationAttempt? = null
                var attemptsUsed = 0
                for ((attemptIndex, budget) in budgets.withIndex()) {
                    val (contexts, droppedBlocks) = packGenerationContexts(
                        item.question, patientIds, candidatesByPatient,
                        budget.answerReserve, budget.charsPerToken, budget.contextCharCap
                    )
                    val evalPrompt = buildFactualQaPrompt(item.question, contexts)
                    val estTokens = estimateTokens(evalPrompt, budget.charsPerToken)
                    Log.i(
                        TAG,
                        "[${item.id}] generationInput attempt=${attemptIndex + 1}/${budgets.size} " +
                            "numPatients=${patientIds.size} seqLen=$READER_MAX_SEQ_LEN " +
                            "answerReserve=${budget.answerReserve} promptChars=${evalPrompt.length} " +
                            "estPromptTokens=$estTokens blocks=${contexts.size} dropped=$droppedBlocks"
                    )
                    val attempt = generateOnce(module, evalPrompt, contexts.size, droppedBlocks, item.id)
                    attemptsUsed++
                    finalAttempt = attempt
                    val clean = attempt.answer.isNotBlank() && attempt.error == null && !attempt.likelyTruncated
                    Log.i(
                        TAG,
                        "[${item.id}] generationResult attempt=$attemptsUsed clean=$clean " +
                            "blank=${attempt.answer.isBlank()} error=${attempt.error} " +
                            "promptTokens=${attempt.promptTokens} generatedTokens=${attempt.generatedTokens} " +
                            "likelyTruncated=${attempt.likelyTruncated} stop=${attempt.sawChatmlStopToken}"
                    )
                    if (clean) break
                }
                val attempt = finalAttempt ?: throw IllegalStateException("[${item.id}] no generation attempt was executed")
                val finalClean = attempt.answer.isNotBlank() && attempt.error == null && !attempt.likelyTruncated
                if (!finalClean) {
                    throw IllegalStateException(
                        "[${item.id}] reader generation remained invalid after $attemptsUsed attempts: " +
                            "blank=${attempt.answer.isBlank()} error=${attempt.error} " +
                            "likelyTruncated=${attempt.likelyTruncated} promptTokens=${attempt.promptTokens} " +
                            "generatedTokens=${attempt.generatedTokens} blocks=${attempt.contextBlocks} dropped=${attempt.droppedBlocks}"
                    )
                }
                generatedAnswer = attempt.answer
                generationDiagnostics = GenerationDiagnostics(
                    attempt.promptChars, attempt.contextBlocks, attempt.droppedBlocks,
                    attemptsUsed > 1, attemptsUsed, attempt.promptTokens, attempt.generatedTokens,
                    attempt.likelyTruncated, attempt.sawChatmlStopToken, attempt.statsText, attempt.error
                )
            }

            val result = ItemResult(
                id = item.id,
                comparisonType = item.comparisonType,
                numSources = numSources,
                numPatients = patientIds.size,
                sourceHitAt1 = sourceHitAt1,
                sourceHitAt3 = sourceHitAt3,
                sourceHitAt5 = sourceHitAt5,
                sourceHitAt10 = sourceHitAt10,
                allSourcesHitAt1 = allSourcesHitAt1,
                allSourcesHitAt3 = allSourcesHitAt3,
                allSourcesHitAt5 = allSourcesHitAt5,
                allSourcesHitAt10 = allSourcesHitAt10,
                sourceMrrMean = sourceMrrMean,
                sourceMrrWorst = sourceMrrWorst,
                sourcePrecisionAt3 = precisionAt3Values.average(),
                sourcePrecisionAt5 = precisionAt5Values.average(),
                sourcePrecisionAt10 = precisionAt10Values.average(),
                sourceNdcgAt3 = ndcgAt3Values.average(),
                sourceNdcgAt10 = ndcgAt10Values.average(),
                sourceAveragePrecision = averagePrecisionValues.average(),
                wrongAdmissionRateAt3 = wrongAdmissionRateAt3,
                wrongAdmissionRateAt10 = wrongAdmissionRateAt10,
                generatedAnswer = generatedAnswer,
                semanticSimilarity = semanticSimilarity,
                generationPass = generationPass,
                generationDiagnostics = generationDiagnostics
            )
            results.add(result)

            Log.i(
                TAG,
                "[${index + 1}/${items.size}] [${result.id}] type=${result.comparisonType} " +
                    "nSources=${result.numSources} nPatients=${result.numPatients} " +
                    "sourceHit@3=%.2f allHit@3=%s mrrMean=%.2f mrrWorst=%.2f wrongAdm@3=%s semSim=%s".format(
                        result.sourceHitAt3, result.allSourcesHitAt3, result.sourceMrrMean, result.sourceMrrWorst,
                        result.wrongAdmissionRateAt3?.let { "%.2f".format(it) } ?: "n/a",
                        result.semanticSimilarity?.let { "%.2f".format(it) } ?: "n/a"
                    )
            )
        }

        val n = results.size
        val meanSourceHitAt1 = results.sumOf { it.sourceHitAt1 } / n
        val meanSourceHitAt3 = results.sumOf { it.sourceHitAt3 } / n
        val meanSourceHitAt5 = results.sumOf { it.sourceHitAt5 } / n
        val meanSourceHitAt10 = results.sumOf { it.sourceHitAt10 } / n
        val allSourcesHitAt1Rate = results.count { it.allSourcesHitAt1 }.toDouble() / n
        val allSourcesHitAt3Rate = results.count { it.allSourcesHitAt3 }.toDouble() / n
        val allSourcesHitAt5Rate = results.count { it.allSourcesHitAt5 }.toDouble() / n
        val allSourcesHitAt10Rate = results.count { it.allSourcesHitAt10 }.toDouble() / n
        val meanSourceMrr = results.sumOf { it.sourceMrrMean } / n
        val meanWorstSourceMrr = results.sumOf { it.sourceMrrWorst } / n
        val meanSourcePrecisionAt3 = results.sumOf { it.sourcePrecisionAt3 } / n
        val meanSourcePrecisionAt5 = results.sumOf { it.sourcePrecisionAt5 } / n
        val meanSourcePrecisionAt10 = results.sumOf { it.sourcePrecisionAt10 } / n
        val meanSourceNdcgAt3 = results.sumOf { it.sourceNdcgAt3 } / n
        val meanSourceNdcgAt10 = results.sumOf { it.sourceNdcgAt10 } / n
        val meanSourceMap = results.sumOf { it.sourceAveragePrecision } / n
        val scoredWrongAdmAt3 = results.mapNotNull { it.wrongAdmissionRateAt3 }
        val meanWrongAdmissionRateAt3 = if (scoredWrongAdmAt3.isNotEmpty()) scoredWrongAdmAt3.average() else null
        val scoredWrongAdmAt10 = results.mapNotNull { it.wrongAdmissionRateAt10 }
        val meanWrongAdmissionRateAt10 = if (scoredWrongAdmAt10.isNotEmpty()) scoredWrongAdmAt10.average() else null

        val scoredGeneration = results.filter { it.semanticSimilarity != null }
        val avgSemanticSimilarity = if (scoredGeneration.isNotEmpty()) {
            scoredGeneration.sumOf { it.semanticSimilarity!! } / scoredGeneration.size
        } else null
        val generationPassRate = if (scoredGeneration.isNotEmpty()) {
            scoredGeneration.count { it.generationPass == true }.toDouble() / scoredGeneration.size
        } else null

        Log.i(
            TAG,
            ("SUMMARY: n=$n source_hit@1=%.1f%% source_hit@3=%.1f%% source_hit@5=%.1f%% source_hit@10=%.1f%% " +
                "all_sources_hit@1=%.1f%% all_sources_hit@3=%.1f%% all_sources_hit@5=%.1f%% all_sources_hit@10=%.1f%% " +
                "mean_source_mrr=%.3f mean_worst_source_mrr=%.3f wrong_adm@3=%s wrong_adm@10=%s avg_sem_sim=%s pass_rate=%s").format(
                meanSourceHitAt1 * 100, meanSourceHitAt3 * 100, meanSourceHitAt5 * 100, meanSourceHitAt10 * 100,
                allSourcesHitAt1Rate * 100, allSourcesHitAt3Rate * 100, allSourcesHitAt5Rate * 100, allSourcesHitAt10Rate * 100,
                meanSourceMrr, meanWorstSourceMrr,
                meanWrongAdmissionRateAt3?.let { "%.2f".format(it) } ?: "n/a",
                meanWrongAdmissionRateAt10?.let { "%.2f".format(it) } ?: "n/a",
                avgSemanticSimilarity?.let { "%.1f%%".format(it * 100) } ?: "n/a (generation skipped)",
                generationPassRate?.let { "%.1f%%".format(it * 100) } ?: "n/a (generation skipped)"
            )
        )

        for (type in results.map { it.comparisonType }.distinct()) {
            val subset = results.filter { it.comparisonType == type }
            val subN = subset.size
            val subHit3 = subset.sumOf { it.sourceHitAt3 } / subN
            val subMrrWorst = subset.sumOf { it.sourceMrrWorst } / subN
            val subWrongAdm3 = subset.mapNotNull { it.wrongAdmissionRateAt3 }.takeIf { it.isNotEmpty() }?.average()
            val subScored = subset.filter { it.semanticSimilarity != null }
            val subPassRate = if (subScored.isNotEmpty()) subScored.count { it.generationPass == true }.toDouble() / subScored.size else null
            Log.i(
                TAG,
                "  [$type] n=$subN source_hit@3=%.1f%% worst_source_mrr=%.3f wrong_adm@3=%s pass_rate=%s".format(
                    subHit3 * 100, subMrrWorst,
                    subWrongAdm3?.let { "%.2f".format(it) } ?: "n/a",
                    subPassRate?.let { "%.1f%%".format(it * 100) } ?: "n/a"
                )
            )
        }

        val generationAttempted = results.count { it.generationDiagnostics != null }
        val generationRetried = results.count { (it.generationDiagnostics?.attemptsUsed ?: 0) > 1 }
        val generationFinalBlank = results.count { it.generationDiagnostics != null && it.generatedAnswer.isNullOrBlank() }
        val generationFinalTruncated = results.count { it.generationDiagnostics?.likelyTruncated == true }
        val generationFinalErrors = results.count { it.generationDiagnostics?.error != null }
        Log.i(
            TAG,
            "GENERATION HEALTH: attempted=$generationAttempted judgeable=${generationAttempted - generationFinalBlank - generationFinalTruncated - generationFinalErrors} " +
                "retried=$generationRetried blank=$generationFinalBlank truncated=$generationFinalTruncated errors=$generationFinalErrors"
        )

        writeResultsJson(
            datasetAsset, results, n, meanSourceHitAt1, meanSourceHitAt3, meanSourceHitAt5, meanSourceHitAt10,
            allSourcesHitAt1Rate, allSourcesHitAt3Rate, allSourcesHitAt5Rate, allSourcesHitAt10Rate,
            meanSourceMrr, meanWorstSourceMrr,
            meanSourcePrecisionAt3, meanSourcePrecisionAt5, meanSourcePrecisionAt10,
            meanSourceNdcgAt3, meanSourceNdcgAt10, meanSourceMap,
            meanWrongAdmissionRateAt3, meanWrongAdmissionRateAt10,
            avgSemanticSimilarity, generationPassRate
        )

        assertTrue("No items were scored", results.isNotEmpty())
    }

    private fun writeResultsJson(
        datasetAsset: String,
        results: List<ItemResult>,
        n: Int,
        meanSourceHitAt1: Double,
        meanSourceHitAt3: Double,
        meanSourceHitAt5: Double,
        meanSourceHitAt10: Double,
        allSourcesHitAt1Rate: Double,
        allSourcesHitAt3Rate: Double,
        allSourcesHitAt5Rate: Double,
        allSourcesHitAt10Rate: Double,
        meanSourceMrr: Double,
        meanWorstSourceMrr: Double,
        meanSourcePrecisionAt3: Double,
        meanSourcePrecisionAt5: Double,
        meanSourcePrecisionAt10: Double,
        meanSourceNdcgAt3: Double,
        meanSourceNdcgAt10: Double,
        meanSourceMap: Double,
        meanWrongAdmissionRateAt3: Double?,
        meanWrongAdmissionRateAt10: Double?,
        avgSemanticSimilarity: Double?,
        generationPassRate: Double?
    ) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val datasetStem = datasetAsset.substringAfterLast('/').substringBeforeLast('.')
        val outFileName = "rrf_bm25_multidoc_rag_results_$datasetStem.json"

        val additionalOutputDir = InstrumentationRegistry.getArguments().getString("additionalTestOutputDir")
        val outFile = if (additionalOutputDir != null) {
            File(additionalOutputDir, outFileName)
        } else {
            File(context.getExternalFilesDir(null), outFileName)
        }

        val root = JSONObject()
        root.put("num_items", n)
        root.put("source_hit_rate_at_1", meanSourceHitAt1)
        root.put("source_hit_rate_at_3", meanSourceHitAt3)
        root.put("source_hit_rate_at_5", meanSourceHitAt5)
        root.put("source_hit_rate_at_10", meanSourceHitAt10)
        root.put("all_sources_hit_rate_at_1", allSourcesHitAt1Rate)
        root.put("all_sources_hit_rate_at_3", allSourcesHitAt3Rate)
        root.put("all_sources_hit_rate_at_5", allSourcesHitAt5Rate)
        root.put("all_sources_hit_rate_at_10", allSourcesHitAt10Rate)
        root.put("mean_source_mrr", meanSourceMrr)
        root.put("mean_worst_source_mrr", meanWorstSourceMrr)
        root.put("mean_source_precision_at_3", meanSourcePrecisionAt3)
        root.put("mean_source_precision_at_5", meanSourcePrecisionAt5)
        root.put("mean_source_precision_at_10", meanSourcePrecisionAt10)
        root.put("mean_source_ndcg_at_3", meanSourceNdcgAt3)
        root.put("mean_source_ndcg_at_10", meanSourceNdcgAt10)
        root.put("mean_source_map", meanSourceMap)
        root.put("mean_wrong_admission_rate_at_3", meanWrongAdmissionRateAt3 ?: JSONObject.NULL)
        root.put("mean_wrong_admission_rate_at_10", meanWrongAdmissionRateAt10 ?: JSONObject.NULL)
        root.put("generation_evaluated", avgSemanticSimilarity != null)
        root.put("avg_semantic_similarity", avgSemanticSimilarity ?: JSONObject.NULL)
        root.put("generation_pass_rate", generationPassRate ?: JSONObject.NULL)
        root.put("reader_max_seq_len", READER_MAX_SEQ_LEN)
        root.put("generation_candidate_top_k_per_patient", GENERATION_CANDIDATE_TOP_K_PER_PATIENT)

        val itemsJson = JSONArray()
        for (r in results) {
            val itemJson = JSONObject()
            itemJson.put("id", r.id)
            itemJson.put("comparison_type", r.comparisonType)
            itemJson.put("num_sources", r.numSources)
            itemJson.put("num_patients", r.numPatients)
            itemJson.put("source_hit_at_1", r.sourceHitAt1)
            itemJson.put("source_hit_at_3", r.sourceHitAt3)
            itemJson.put("source_hit_at_5", r.sourceHitAt5)
            itemJson.put("source_hit_at_10", r.sourceHitAt10)
            itemJson.put("all_sources_hit_at_1", r.allSourcesHitAt1)
            itemJson.put("all_sources_hit_at_3", r.allSourcesHitAt3)
            itemJson.put("all_sources_hit_at_5", r.allSourcesHitAt5)
            itemJson.put("all_sources_hit_at_10", r.allSourcesHitAt10)
            itemJson.put("source_mrr_mean", r.sourceMrrMean)
            itemJson.put("source_mrr_worst", r.sourceMrrWorst)
            itemJson.put("source_precision_at_3", r.sourcePrecisionAt3)
            itemJson.put("source_precision_at_5", r.sourcePrecisionAt5)
            itemJson.put("source_precision_at_10", r.sourcePrecisionAt10)
            itemJson.put("source_ndcg_at_3", r.sourceNdcgAt3)
            itemJson.put("source_ndcg_at_10", r.sourceNdcgAt10)
            itemJson.put("source_average_precision", r.sourceAveragePrecision)
            itemJson.put("wrong_admission_rate_at_3", r.wrongAdmissionRateAt3 ?: JSONObject.NULL)
            itemJson.put("wrong_admission_rate_at_10", r.wrongAdmissionRateAt10 ?: JSONObject.NULL)
            itemJson.put("generated_answer", r.generatedAnswer ?: JSONObject.NULL)
            itemJson.put("semantic_similarity", r.semanticSimilarity ?: JSONObject.NULL)
            itemJson.put("generation_pass", r.generationPass ?: JSONObject.NULL)
            val gd = r.generationDiagnostics
            itemJson.put("generation_prompt_chars", gd?.promptChars ?: JSONObject.NULL)
            itemJson.put("generation_context_blocks", gd?.contextBlocks ?: JSONObject.NULL)
            itemJson.put("generation_dropped_blocks", gd?.droppedBlocks ?: JSONObject.NULL)
            itemJson.put("generation_retry_used", gd?.retryUsed ?: JSONObject.NULL)
            itemJson.put("generation_attempts_used", gd?.attemptsUsed ?: JSONObject.NULL)
            itemJson.put("generation_prompt_tokens", gd?.promptTokens ?: JSONObject.NULL)
            itemJson.put("generation_generated_tokens", gd?.generatedTokens ?: JSONObject.NULL)
            itemJson.put("generation_likely_truncated", gd?.likelyTruncated ?: JSONObject.NULL)
            itemJson.put("generation_saw_chatml_stop", gd?.sawChatmlStopToken ?: JSONObject.NULL)
            itemJson.put("generation_stats_text", gd?.statsText ?: JSONObject.NULL)
            itemJson.put("generation_error", gd?.error ?: JSONObject.NULL)
            itemsJson.put(itemJson)
        }
        root.put("items", itemsJson)

        outFile.writeText(root.toString(2))
        Log.i(TAG, "Results written to ${outFile.path}")
    }
}
