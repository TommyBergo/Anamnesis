/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo.rag

import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.rag.data.ScoredChunk
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Locale

/** Verifies the chunk-ranking logic mirrored from [MedicalRagEngine].preparePrompt: given arbitrary input order, chunk #1 must always be the highest-scoring result and chunk #2 the second-highest. */
class MedicalRagEngineRankingTest {

    private fun chunk(id: Long, content: String) = ClinicalChunk(
        id = id,
        patientId = "default_patient",
        documentType = "Psychiatric Medical Record",
        documentId = "doc1",
        content = content,
        dateString = "2026-07-30"
    )

    private fun rankAndBound(raw: List<ScoredChunk>, maxContextCharsPerChunk: Int = 400): List<ScoredChunk> {
        val sorted = raw.sortedByDescending { it.score }
        return sorted.map { scored ->
            val content = scored.chunk.content
            if (content.length <= maxContextCharsPerChunk) {
                scored
            } else {
                scored.copy(chunk = scored.chunk.copy(content = content.take(maxContextCharsPerChunk) + "..."))
            }
        }
    }

    private fun buildHeader(boundedChunks: List<ScoredChunk>): String {
        val sb = StringBuilder()
        boundedChunks.forEachIndexed { index, scored ->
            val scoreFormatted = String.format(Locale.ROOT, "%.1f%%", scored.score)
            sb.append("[Chunk #${index + 1}] Coherence: $scoreFormatted\n")
        }
        return sb.toString()
    }

    @Test
    fun `chunk 1 is always the highest score, regardless of input order`() {
        val raw = listOf(
            ScoredChunk(chunk(3, "middling match"), score = 55.8f),
            ScoredChunk(chunk(1, "best match"), score = 90.2f),
            ScoredChunk(chunk(5, "worst match"), score = 12.3f),
            ScoredChunk(chunk(2, "second best match"), score = 67.9f),
            ScoredChunk(chunk(4, "another middling match"), score = 33.1f)
        )

        val ranked = rankAndBound(raw)

        assertEquals("Chunk #1 must be the globally highest score", 90.2f, ranked.first().score)
        assertEquals("best match", ranked.first().chunk.content)
        assertEquals("Chunk #2 must be the second-highest score", 67.9f, ranked[1].score)
        assertEquals("second best match", ranked[1].chunk.content)
    }

    @Test
    fun `full ranking is strictly non-increasing by score`() {
        val raw = listOf(
            ScoredChunk(chunk(1, "a"), score = 40.0f),
            ScoredChunk(chunk(2, "b"), score = 95.5f),
            ScoredChunk(chunk(3, "c"), score = 61.2f),
            ScoredChunk(chunk(4, "d"), score = 61.2f),
            ScoredChunk(chunk(5, "e"), score = 5.0f)
        )

        val ranked = rankAndBound(raw)

        for (i in 0 until ranked.size - 1) {
            assertTrue(
                "Chunk #${i + 1} (${ranked[i].score}) must be >= Chunk #${i + 2} (${ranked[i + 1].score})",
                ranked[i].score >= ranked[i + 1].score
            )
        }
    }

    @Test
    fun `header labels line up with descending score order, not insertion order`() {
        val raw = listOf(
            ScoredChunk(chunk(1, "inserted first, but a poor match"), score = 20.0f),
            ScoredChunk(chunk(2, "inserted second, but the best match"), score = 88.0f)
        )

        val header = buildHeader(rankAndBound(raw))

        val chunk1Line = header.lines().first { it.startsWith("[Chunk #1]") }
        val chunk2Line = header.lines().first { it.startsWith("[Chunk #2]") }
        assertTrue("Chunk #1 header must report the higher score (88.0%)", chunk1Line.contains("88.0%"))
        assertTrue("Chunk #2 header must report the lower score (20.0%)", chunk2Line.contains("20.0%"))
    }

    @Test
    fun `truncating long chunk content never changes the ranking order`() {
        val longContent = "x".repeat(1000)
        val raw = listOf(
            ScoredChunk(chunk(1, longContent), score = 50.0f),
            ScoredChunk(chunk(2, "short"), score = 99.0f)
        )

        val ranked = rankAndBound(raw, maxContextCharsPerChunk = 400)

        assertEquals(99.0f, ranked.first().score)
        assertEquals("short", ranked.first().chunk.content)
        assertTrue("Long content must be truncated, not dropped", ranked[1].chunk.content.endsWith("..."))
        assertEquals(400 + 3, ranked[1].chunk.content.length)
    }
}
