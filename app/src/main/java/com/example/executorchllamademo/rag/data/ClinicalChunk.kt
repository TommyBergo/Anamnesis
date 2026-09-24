package com.example.executorchllamademo.rag.data

import io.objectbox.annotation.Entity
import io.objectbox.annotation.HnswIndex
import io.objectbox.annotation.Id

const val CLINICAL_EMBEDDING_DIMENSION = 768

/** Represents a single retrievable clinical text chunk with its embedding vector, stored as an ObjectBox entity. */
@Entity
data class ClinicalChunk(
    @Id var id: Long = 0,
    var patientId: String = "",
    var documentType: String = "",
    var documentId: String = "",
    var content: String = "",
    var dateString: String = "",
    var timestamp: Long = System.currentTimeMillis(),

    @HnswIndex(dimensions = 768)
    var embedding: FloatArray? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as ClinicalChunk
        return id == other.id
    }

    override fun hashCode(): Int = id.hashCode()
}