package com.example.executorchllamademo.rag

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.ContextCompat
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.rag.data.MyObjectBox
import com.example.executorchllamademo.rag.data.ScoredChunk
import com.example.executorchllamademo.rag.embedding.Embedder
import com.example.executorchllamademo.rag.embedding.GraniteEmbedder
import com.example.executorchllamademo.rag.embedding.EmbeddingGemmaEmbedder
import com.example.executorchllamademo.rag.repository.BruteForceVectorRepository
import com.example.executorchllamademo.rag.repository.MedicalVectorRepository
import com.example.executorchllamademo.rag.repository.SqliteVecVectorRepository
import com.example.executorchllamademo.rag.repository.UsearchVectorRepository
import io.objectbox.BoxStore
import java.io.Closeable
import java.io.File
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/** Compares embedded vector stores (ObjectBox HNSW, sqlite-vec, USearch, and a brute-force baseline) for the on-device RAG pipeline by index size, indexing latency, query latency, and recall@10 over the pooled corpus. */
@RunWith(AndroidJUnit4::class)
class VectorStoreBenchmarkTest {

    companion object {
        private const val TAG = "VectorStoreBenchmarkTest"
        private const val ASSET_DIR = "rag_benchmark"
        private const val DEFAULT_DATASET_ASSET = "QA/single_dataset_rulebased.jsonl"
        private const val DEFAULT_PDF_SUBDIR = "Ammissioni_PDF_stratified_enriched"
        private const val TEST_DB_NAME = "objectbox-vector-store-benchmark"
        private const val TOP_K = 10

        private val ADMISSION_ID_REGEX = Regex("Admission_(\\d+)")

        private fun percentile(sortedValues: List<Double>, p: Double): Double {
            if (sortedValues.isEmpty()) return 0.0
            val index = ((p / 100.0) * (sortedValues.size - 1)).toInt().coerceIn(0, sortedValues.size - 1)
            return sortedValues[index]
        }
    }

    private lateinit var embedder: Embedder

    @Before
    fun setUp() {
        val targetContext = InstrumentationRegistry.getInstrumentation().targetContext
        ContextCompat.startForegroundService(
            targetContext,
            Intent(targetContext, BenchmarkKeepAliveService::class.java)
        )

        val embedderBackend = InstrumentationRegistry.getArguments().getString("embedderBackend", "mediapipe") ?: "mediapipe"
        embedder = when (embedderBackend) {
            "mediapipe" -> EmbeddingGemmaEmbedder(targetContext)
            "granite" -> GraniteEmbedder(targetContext)
            else -> throw IllegalArgumentException("Unknown embedderBackend '$embedderBackend'")
        }
        Log.i(TAG, "setUp: embedder ready (backendId=${embedder.backendId}, dim=${embedder.embeddingDimension})")
    }

    @After
    fun tearDown() {
        (embedder as? Closeable)?.close()
        val targetContext = InstrumentationRegistry.getInstrumentation().targetContext
        targetContext.stopService(Intent(targetContext, BenchmarkKeepAliveService::class.java))
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
    fun runVectorStoreBenchmark() {
        val targetContext = InstrumentationRegistry.getInstrumentation().targetContext
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val args = InstrumentationRegistry.getArguments()

        val store = args.getString("store", "objectbox") ?: "objectbox"
        val chunkSize = args.getString("chunkSize", "")?.toIntOrNull() ?: 500
        val overlapChars = args.getString("overlapChars", "")?.toIntOrNull() ?: 100
        val maxItems = args.getString("maxItems", "")?.toIntOrNull()
        val datasetAsset = "$ASSET_DIR/" + (args.getString("datasetAsset", "") ?: "").ifBlank { DEFAULT_DATASET_ASSET }
        val pdfAssetDir = "$ASSET_DIR/" + (args.getString("pdfAssetDir", "") ?: "").ifBlank { DEFAULT_PDF_SUBDIR }

        val items = testContext.assets.open(datasetAsset).bufferedReader().use { it.readLines() }
            .filter { it.isNotBlank() }
            .map { JSONObject(it) }
            .let { if (maxItems != null) it.take(maxItems) else it }
        assertTrue("Dataset asset was empty", items.isNotEmpty())
        Log.i(
            TAG,
            "Loaded ${items.size} dataset items from $datasetAsset (store=$store, chunkSize=$chunkSize, " +
                "overlapChars=$overlapChars, embedder=${embedder.backendId})"
        )

        data class DocKey(val patientId: String, val documentName: String)
        val distinctDocs = items.map { DocKey(it.getString("patient_id"), it.getString("document_name")) }.distinct()

        val pdfProcessor = ClinicalPdfProcessor(targetContext)
        val allChunks = mutableListOf<ClinicalChunk>()
        for ((index, doc) in distinctDocs.withIndex()) {
            val pdfFile = copyAssetToCache("$pdfAssetDir/${doc.documentName}", targetContext)
            val rawText = runBlocking { pdfProcessor.extractText(Uri.fromFile(pdfFile)) }
            val chunks = pdfProcessor.createClinicalChunks(rawText, maxChars = chunkSize, overlapChars = overlapChars)
            val admissionId = ADMISSION_ID_REGEX.find(doc.documentName)?.groupValues?.get(1) ?: doc.documentName
            for (chunkText in chunks) {
                allChunks.add(
                    ClinicalChunk(
                        patientId = doc.patientId,
                        documentType = "Clinical Discharge Summary",
                        documentId = admissionId,
                        content = chunkText
                    ).apply { embedding = embedder.generateEmbedding(chunkText) }
                )
            }
            pdfFile.delete()
            if ((index + 1) % 20 == 0 || index == distinctDocs.size - 1) {
                Log.i(TAG, "[${index + 1}/${distinctDocs.size}] Prepared ${doc.documentName}: ${chunks.size} chunks (running total ${allChunks.size})")
            }
        }
        assertTrue("No chunks produced from the corpus", allChunks.isNotEmpty())
        Log.i(TAG, "Corpus ready: ${allChunks.size} chunks from ${distinctDocs.size} documents")

        if (store != "objectbox") {
            allChunks.forEachIndexed { index, chunk -> chunk.id = index + 1L }
        }

        var objectBoxRepo: MedicalVectorRepository? = null
        var boxStore: BoxStore? = null
        var sqliteVecRepo: SqliteVecVectorRepository? = null
        var bruteForceRepo: BruteForceVectorRepository? = null
        var usearchRepo: UsearchVectorRepository? = null

        val indexingStartMs = System.currentTimeMillis()
        when (store) {
            "objectbox" -> {
                BoxStore.deleteAllFiles(targetContext.applicationContext, TEST_DB_NAME)
                boxStore = MyObjectBox.builder()
                    .androidContext(targetContext.applicationContext)
                    .name(TEST_DB_NAME)
                    .build()
                objectBoxRepo = MedicalVectorRepository(boxStore).also { it.insertChunks(allChunks) }
            }
            "sqlitevec" -> {
                sqliteVecRepo = SqliteVecVectorRepository(targetContext, dimensions = embedder.embeddingDimension)
                    .also { it.insertChunks(allChunks) }
            }
            "bruteforce" -> {
                bruteForceRepo = BruteForceVectorRepository(targetContext, dimensions = embedder.embeddingDimension)
                    .also { it.insertChunks(allChunks) }
            }
            "usearch" -> {
                usearchRepo = UsearchVectorRepository(targetContext, dimensions = embedder.embeddingDimension)
                    .also { it.insertChunks(allChunks) }
            }
            else -> throw IllegalArgumentException("Unknown store '$store' - expected objectbox|sqlitevec|bruteforce|usearch")
        }
        val indexingLatencyMs = System.currentTimeMillis() - indexingStartMs

        val groundTruth = BruteForceVectorRepository(targetContext, dimensions = embedder.embeddingDimension)
        groundTruth.insertChunks(allChunks)

        @Suppress("DEPRECATION")
        val indexSizeBytes = when (store) {
            "objectbox" -> boxStore?.sizeOnDisk() ?: 0L
            "sqlitevec" -> sqliteVecRepo?.diskSizeBytes() ?: 0L
            "bruteforce" -> bruteForceRepo?.diskSizeBytes() ?: 0L
            "usearch" -> usearchRepo?.diskSizeBytes() ?: 0L
            else -> 0L
        }

        val queryLatenciesMs = mutableListOf<Double>()
        val recalls = mutableListOf<Double>()
        for ((index, item) in items.withIndex()) {
            val question = item.getString("question")
            val queryEmbedding = embedder.generateEmbedding(question, isQuery = true)

            val startNs = System.nanoTime()
            val retrieved: List<ScoredChunk> = when (store) {
                "objectbox" -> objectBoxRepo!!.searchSimilarChunksWithScore(queryEmbedding, patientId = null, topK = TOP_K)
                "sqlitevec" -> sqliteVecRepo!!.searchTopK(queryEmbedding, TOP_K)
                "bruteforce" -> bruteForceRepo!!.searchTopK(queryEmbedding, TOP_K)
                "usearch" -> usearchRepo!!.searchTopK(queryEmbedding, TOP_K)
                else -> emptyList()
            }
            val elapsedMs = (System.nanoTime() - startNs) / 1_000_000.0
            queryLatenciesMs.add(elapsedMs)

            val exactIds = groundTruth.searchTopK(queryEmbedding, TOP_K).map { it.chunk.id }.toSet()
            val retrievedIds = retrieved.map { it.chunk.id }.toSet()
            val recall = if (exactIds.isEmpty()) 0.0 else retrievedIds.count { it in exactIds }.toDouble() / exactIds.size
            recalls.add(recall)

            if ((index + 1) % 20 == 0 || index == items.size - 1) {
                Log.i(TAG, "[${index + 1}/${items.size}] query latency=%.3fms recall@10=%.2f".format(elapsedMs, recall))
            }
        }

        val sortedLatencies = queryLatenciesMs.sorted()
        val p50 = percentile(sortedLatencies, 50.0)
        val p95 = percentile(sortedLatencies, 95.0)
        val meanRecall = recalls.average()

        Log.i(
            TAG,
            "SUMMARY: store=$store embedder=${embedder.backendId} n_chunks=${allChunks.size} " +
                "n_queries=${items.size} index_size_bytes=$indexSizeBytes indexing_latency_ms=$indexingLatencyMs " +
                "query_p50_ms=%.3f query_p95_ms=%.3f recall@10=%.3f".format(p50, p95, meanRecall)
        )

        writeResultsJson(
            store = store,
            embedderBackend = embedder.backendId,
            numChunks = allChunks.size,
            numQueries = items.size,
            indexSizeBytes = indexSizeBytes,
            indexingLatencyMs = indexingLatencyMs,
            queryP50Ms = p50,
            queryP95Ms = p95,
            meanRecallAt10 = meanRecall,
            queryLatenciesMs = queryLatenciesMs
        )

        objectBoxRepo?.let {
            boxStore?.close()
            BoxStore.deleteAllFiles(targetContext.applicationContext, TEST_DB_NAME)
        }
        sqliteVecRepo?.close()
        bruteForceRepo?.close()
        usearchRepo?.close()
        groundTruth.close()

        assertTrue("No queries were scored", queryLatenciesMs.isNotEmpty())
    }

    private fun writeResultsJson(
        store: String,
        embedderBackend: String,
        numChunks: Int,
        numQueries: Int,
        indexSizeBytes: Long,
        indexingLatencyMs: Long,
        queryP50Ms: Double,
        queryP95Ms: Double,
        meanRecallAt10: Double,
        queryLatenciesMs: List<Double>
    ) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val outFileName = "vector_store_benchmark_$store.json"

        val additionalOutputDir = InstrumentationRegistry.getArguments().getString("additionalTestOutputDir")
        val outFile = if (additionalOutputDir != null) {
            File(additionalOutputDir, outFileName)
        } else {
            File(context.getExternalFilesDir(null), outFileName)
        }

        val root = JSONObject()
        root.put("store", store)
        root.put("embedder_backend", embedderBackend)
        root.put("num_chunks", numChunks)
        root.put("num_queries", numQueries)
        root.put("index_size_bytes", indexSizeBytes)
        root.put("indexing_latency_ms", indexingLatencyMs)
        root.put("query_latency_ms_p50", queryP50Ms)
        root.put("query_latency_ms_p95", queryP95Ms)
        root.put("recall_at_10", meanRecallAt10)
        root.put("query_latencies_ms", JSONArray(queryLatenciesMs))

        outFile.writeText(root.toString(2))
        Log.i(TAG, "Results written to ${outFile.path}")
    }
}
