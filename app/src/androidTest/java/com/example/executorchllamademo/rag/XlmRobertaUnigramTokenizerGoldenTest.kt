package com.example.executorchllamademo.rag

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.executorchllamademo.rag.embedding.XlmRobertaUnigramTokenizer
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Verifies [XlmRobertaUnigramTokenizer] against real HF `tokenizers` output on golden fixtures, requiring 100% exact match on edge cases and >=99.5% on real-corpus text. */
@RunWith(AndroidJUnit4::class)
class XlmRobertaUnigramTokenizerGoldenTest {

    companion object {
        private const val TAG = "XlmRobertaUnigramGolden"
        private const val EDGE_CASE_COUNT = 12
        private const val REAL_CORPUS_PASS_THRESHOLD = 0.995
    }

    @Test
    fun testGteMultilingualTokenizerMatchesRealHfTokenizer() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val tokenizer = XlmRobertaUnigramTokenizer(context, "gte_multilingual_tokenizer.json")

        val lines = context.assets.open("tokenizer_golden/gte_multilingual_unigram_golden.jsonl")
            .bufferedReader(Charsets.UTF_8).readLines().filter { it.isNotBlank() }
        assertTrue("Golden fixture file must not be empty", lines.isNotEmpty())

        var edgeCaseFailures = 0
        var realCorpusFailures = 0
        var realCorpusTotal = 0

        lines.forEachIndexed { index, line ->
            val row = JSONObject(line)
            val text = row.getString("text")
            val expectedIds = row.getJSONArray("ids").let { arr ->
                LongArray(arr.length()) { i -> arr.getLong(i) }
            }

            val actual = tokenizer.encode(text, expectedIds.size)
            val matches = actual.ids.contentEquals(expectedIds)

            if (index < EDGE_CASE_COUNT) {
                if (!matches) {
                    edgeCaseFailures++
                    Log.e(TAG, "EDGE CASE MISMATCH [$index] text=${text.take(80)!!}")
                    logFirstDivergence(expectedIds, actual.ids)
                }
            } else {
                realCorpusTotal++
                if (!matches) {
                    realCorpusFailures++
                    Log.w(TAG, "real-corpus mismatch [$index] text=${text.take(80)}")
                    logFirstDivergence(expectedIds, actual.ids)
                }
            }
        }

        val realCorpusPassRate = if (realCorpusTotal == 0) 1.0 else
            (realCorpusTotal - realCorpusFailures).toDouble() / realCorpusTotal
        Log.i(
            TAG,
            "edgeCaseFailures=$edgeCaseFailures/$EDGE_CASE_COUNT " +
                "realCorpusFailures=$realCorpusFailures/$realCorpusTotal " +
                "realCorpusPassRate=$realCorpusPassRate"
        )

        assertTrue(
            "All $EDGE_CASE_COUNT edge cases must match exactly, but $edgeCaseFailures failed",
            edgeCaseFailures == 0
        )
        assertTrue(
            "Real-corpus exact-match rate $realCorpusPassRate must be >= $REAL_CORPUS_PASS_THRESHOLD",
            realCorpusPassRate >= REAL_CORPUS_PASS_THRESHOLD
        )
    }

    private fun logFirstDivergence(expected: LongArray, actual: LongArray) {
        val firstDiff = expected.indices.firstOrNull { it >= actual.size || expected[it] != actual[it] }
        if (firstDiff != null) {
            val expWindow = expected.toList().subList(
                (firstDiff - 2).coerceAtLeast(0), (firstDiff + 3).coerceAtMost(expected.size)
            )
            val actWindow = actual.toList().subList(
                (firstDiff - 2).coerceAtLeast(0), (firstDiff + 3).coerceAtMost(actual.size)
            )
            Log.e(TAG, "  first divergence at index $firstDiff: expected=$expWindow actual=$actWindow")
        }
    }
}
