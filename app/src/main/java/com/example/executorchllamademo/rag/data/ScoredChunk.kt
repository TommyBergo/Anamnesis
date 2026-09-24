package com.example.executorchllamademo.rag.data

/** Pairs a retrieved [ClinicalChunk] with its similarity/relevance score from a retrieval engine. */
data class ScoredChunk(
    val chunk: ClinicalChunk,
    val score: Float
)