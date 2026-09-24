package com.example.executorchllamademo.rag.embedding

import android.content.Context
import android.util.JsonReader
import android.util.Log
import java.io.IOException
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets
import java.util.Locale

/** Thrown when the Granite tokenizer asset fails to load or parse. */
class GraniteTokenizerException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

private const val TAG = "GraniteTokenizer"

/** Streaming BPE tokenizer for Granite's bundled `tokenizer.json`, producing right-padded token-id and attention-mask arrays for [GraniteEmbedder]. */
class GraniteTokenizer(context: Context, assetFileName: String = "granite_tokenizer.json") {

    class Encoding(val ids: IntArray, val attentionMask: IntArray)

    private val vocab = HashMap<String, Int>()
    private val mergeRanks = HashMap<String, Int>()

    private val padId: Int
    private val bosId: Int
    private val unkId: Int

    init {
        try {
            context.assets.open(assetFileName).use { input ->
                JsonReader(InputStreamReader(input, StandardCharsets.UTF_8)).use { reader ->
                    reader.beginObject()
                    while (reader.hasNext()) {
                        if (reader.nextName() == "model") parseModel(reader) else reader.skipValue()
                    }
                    reader.endObject()
                }
            }
        } catch (e: IOException) {
            throw GraniteTokenizerException("Unable to read tokenizer asset '$assetFileName'", e)
        } catch (e: Exception) {
            throw GraniteTokenizerException("Malformed tokenizer asset '$assetFileName'", e)
        }

        if (vocab.isEmpty()) {
            throw GraniteTokenizerException("Tokenizer vocabulary is empty after parsing '$assetFileName'")
        }

        padId = vocab["<pad>"] ?: 0
        bosId = vocab["<bos>"] ?: padId
        unkId = vocab["<unk>"] ?: padId
    }

    private fun parseModel(reader: JsonReader) {
        reader.beginObject()
        while (reader.hasNext()) {
            when (reader.nextName()) {
                "vocab" -> {
                    reader.beginObject()
                    while (reader.hasNext()) {
                        val token = reader.nextName()
                        vocab[token] = reader.nextInt()
                    }
                    reader.endObject()
                }
                "merges" -> {
                    reader.beginArray()
                    var rank = 0
                    while (reader.hasNext()) {
                        reader.beginArray()
                        val left = reader.nextString()
                        val right = reader.nextString()
                        reader.endArray()
                        mergeRanks[mergeKey(left, right)] = rank++
                    }
                    reader.endArray()
                }
                else -> reader.skipValue()
            }
        }
        reader.endObject()
    }

    private fun mergeKey(left: String, right: String) = "$left$right"

    fun encode(text: String, maxLength: Int): Encoding {
        val ids = ArrayList<Int>(maxLength)
        ids.add(bosId)

        try {
            val encoded = bpeEncodeToIds(text)
            if (encoded.size > maxLength - 1) {
                Log.w(TAG, "Truncated: ${encoded.size} tokens -> ${maxLength - 1} (maxLength=$maxLength)")
            }
            for (id in encoded) {
                if (ids.size >= maxLength) break
                ids.add(id)
            }
        } catch (e: Exception) {
            if (ids.size < maxLength) ids.add(unkId)
        }

        val result = IntArray(maxLength) { padId }
        val attentionMask = IntArray(maxLength)
        for (i in ids.indices) {
            if (i >= maxLength) break
            result[i] = ids[i]
            attentionMask[i] = 1
        }
        return Encoding(result, attentionMask)
    }

    private fun bpeEncodeToIds(text: String): List<Int> {
        val normalized = text.replace(' ', '▁')
        val symbols = normalized.codePoints().toArray()
            .map { String(Character.toChars(it)) }
            .toMutableList()

        while (symbols.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIndex = -1
            for (i in 0 until symbols.size - 1) {
                val rank = mergeRanks[mergeKey(symbols[i], symbols[i + 1])] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestIndex = i
                }
            }
            if (bestIndex == -1) break
            symbols[bestIndex] = symbols[bestIndex] + symbols[bestIndex + 1]
            symbols.removeAt(bestIndex + 1)
        }

        val ids = ArrayList<Int>(symbols.size)
        for (symbol in symbols) {
            val id = vocab[symbol]
            if (id != null) {
                ids.add(id)
            } else {
                for (byte in symbol.toByteArray(StandardCharsets.UTF_8)) {
                    val hexToken = String.format(Locale.ROOT, "<0x%02X>", byte.toInt() and 0xFF)
                    ids.add(vocab[hexToken] ?: unkId)
                }
            }
        }
        return ids
    }
}
