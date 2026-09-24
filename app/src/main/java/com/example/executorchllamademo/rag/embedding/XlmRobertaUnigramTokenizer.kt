package com.example.executorchllamademo.rag.embedding

import android.content.Context
import android.util.JsonReader
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets
import java.text.Normalizer

/** Thrown when the XLM-RoBERTa tokenizer asset fails to load or parse. */
class XlmRobertaUnigramTokenizerException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

private fun openTokenizerFile(context: Context, fileName: String): InputStream {
    val externalFile = File(EXTERNAL_TOKENIZER_DIR, fileName)
    if (externalFile.exists()) {
        return FileInputStream(externalFile)
    }
    return context.assets.open(fileName)
}

private const val EXTERNAL_TOKENIZER_DIR = "/data/local/tmp/llama"

/** SentencePiece Unigram tokenizer, shared across XLM-RoBERTa-family models, that Viterbi-segments text into token ids for [MultilingualE5SmallEmbedder]. */
class XlmRobertaUnigramTokenizer(context: Context, assetFileName: String) {

    class Encoding(val ids: LongArray, val attentionMask: LongArray)

    private val pieceToId = HashMap<String, Int>()
    private var scores = FloatArray(0)
    private var maxPieceCodePoints = 1

    private val padId: Int
    private val bosId: Int
    private val eosId: Int
    private val unkId: Int

    init {
        var unkIdFromFile = 3
        try {
            openTokenizerFile(context, assetFileName).use { input ->
                JsonReader(InputStreamReader(input, StandardCharsets.UTF_8)).use { reader ->
                    reader.beginObject()
                    while (reader.hasNext()) {
                        if (reader.nextName() == "model") {
                            unkIdFromFile = parseModel(reader)
                        } else {
                            reader.skipValue()
                        }
                    }
                    reader.endObject()
                }
            }
        } catch (e: IOException) {
            throw XlmRobertaUnigramTokenizerException("Unable to read tokenizer '$assetFileName'", e)
        } catch (e: Exception) {
            throw XlmRobertaUnigramTokenizerException("Malformed tokenizer '$assetFileName'", e)
        }

        if (pieceToId.isEmpty()) {
            throw XlmRobertaUnigramTokenizerException("Tokenizer vocabulary is empty after parsing '$assetFileName'")
        }

        bosId = pieceToId["<s>"] ?: 0
        padId = pieceToId["<pad>"] ?: bosId
        eosId = pieceToId["</s>"] ?: bosId
        unkId = pieceToId["<unk>"] ?: unkIdFromFile
    }

    private fun parseModel(reader: JsonReader): Int {
        var unkIdFromFile = 3
        val scoreList = ArrayList<Float>(260_000)
        reader.beginObject()
        while (reader.hasNext()) {
            when (reader.nextName()) {
                "unk_id" -> unkIdFromFile = reader.nextInt()
                "vocab" -> {
                    reader.beginArray()
                    var id = 0
                    while (reader.hasNext()) {
                        reader.beginArray()
                        val piece = reader.nextString()
                        val score = reader.nextDouble()
                        reader.endArray()
                        pieceToId[piece] = id
                        scoreList.add(score.toFloat())
                        val pieceLen = piece.codePointCount(0, piece.length)
                        if (pieceLen > maxPieceCodePoints) maxPieceCodePoints = pieceLen
                        id++
                    }
                    reader.endArray()
                }
                else -> reader.skipValue()
            }
        }
        reader.endObject()
        scores = scoreList.toFloatArray()
        return unkIdFromFile
    }

    fun encode(text: String, maxLength: Int): Encoding {
        val bodyIds = try {
            unigramEncodeToIds(text)
        } catch (e: Exception) {
            Log.e(TAG, "Unigram segmentation failed for text of length ${text.length}: ${e.message}", e)
            emptyList()
        }

        val budget = (maxLength - 2).coerceAtLeast(0)
        val truncated = if (bodyIds.size > budget) {
            Log.w(TAG, "Truncated: ${bodyIds.size} tokens -> $budget (maxLength=$maxLength)")
            bodyIds.subList(0, budget)
        } else {
            bodyIds
        }

        val ids = ArrayList<Long>(maxLength)
        ids.add(bosId.toLong())
        for (id in truncated) ids.add(id.toLong())
        if (ids.size < maxLength) ids.add(eosId.toLong())

        val result = LongArray(maxLength) { padId.toLong() }
        val attentionMask = LongArray(maxLength)
        for (i in ids.indices) {
            if (i >= maxLength) break
            result[i] = ids[i]
            attentionMask[i] = 1L
        }
        return Encoding(result, attentionMask)
    }

    private fun normalize(text: String): String {
        val cleaned = text.filter {
            it.code != 0 && it.code != 0xFFFD && !(it.isISOControl() && it != '\t' && it != '\n' && it != '\r')
        }
        val nfkc = Normalizer.normalize(cleaned, Normalizer.Form.NFKC)
        return nfkc.replace(Regex("\\s+"), " ").trim()
    }

    private fun preTokenize(text: String): List<String> {
        if (text.isEmpty()) return emptyList()
        return text.split(WHITESPACE_RUN).filter { it.isNotEmpty() }.map { METASPACE + it }
    }

    private fun unigramEncodeToIds(text: String): List<Int> {
        val normalized = normalize(text)
        val ids = ArrayList<Int>()
        for (word in preTokenize(normalized)) {
            ids.addAll(viterbiSegment(word))
        }
        return ids
    }

    private fun viterbiSegment(word: String): List<Int> {
        val codePoints = word.codePoints().toArray()
        val n = codePoints.size
        if (n == 0) return emptyList()

        val negInf = Double.NEGATIVE_INFINITY
        val best = DoubleArray(n + 1) { negInf }
        val backStart = IntArray(n + 1) { -1 }
        val backId = IntArray(n + 1) { -1 }
        best[0] = 0.0

        val charOffsets = IntArray(n + 1)
        var charIdx = 0
        for (k in 0 until n) {
            charOffsets[k] = charIdx
            charIdx += Character.charCount(codePoints[k])
        }
        charOffsets[n] = charIdx

        for (i in 1..n) {
            val jStart = (i - maxPieceCodePoints).coerceAtLeast(0)
            for (j in jStart until i) {
                if (best[j] == negInf) continue
                val piece = word.substring(charOffsets[j], charOffsets[i])
                val id = pieceToId[piece] ?: continue
                val candidate = best[j] + scores[id]
                if (candidate > best[i]) {
                    best[i] = candidate
                    backStart[i] = j
                    backId[i] = id
                }
            }
            if (best[i] == negInf) {
                val j = i - 1
                best[i] = best[j] + UNK_PENALTY
                backStart[i] = j
                backId[i] = unkId
            }
        }

        val result = ArrayList<Int>()
        var pos = n
        while (pos > 0) {
            result.add(backId[pos])
            pos = backStart[pos]
        }
        result.reverse()
        return result
    }

    companion object {
        private const val TAG = "XlmRobertaUnigramTok"
        private const val METASPACE = "▁"
        private const val UNK_PENALTY = -10000.0
        private val WHITESPACE_RUN = Regex("\\s+")
    }
}
