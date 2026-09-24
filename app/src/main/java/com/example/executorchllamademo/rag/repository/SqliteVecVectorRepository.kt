package com.example.executorchllamademo.rag.repository

import android.content.ContentValues
import android.content.Context
import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.rag.data.ScoredChunk
import io.requery.android.database.sqlite.SQLiteCustomExtension
import io.requery.android.database.sqlite.SQLiteDatabase
import io.requery.android.database.sqlite.SQLiteDatabaseConfiguration
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Vector-store benchmark candidate backed by SQLite plus the sqlite-vec extension, performing exact brute-force distance search via a virtual table. */
class SqliteVecVectorRepository(context: Context, private val dimensions: Int = 768) {

    private val dbFile = File(context.filesDir, "sqlite_vec_benchmark.db")
    private val db: SQLiteDatabase

    init {
        val soFile = File(context.applicationInfo.nativeLibraryDir, "libvec0.so")
        check(soFile.exists()) {
            "libvec0.so not found at ${soFile.absolutePath} - did the sqlite-vec Android build " +
                "get placed under app/src/main/jniLibs/<abi>/libvec0.so before building the APK?"
        }
        dbFile.delete()

        val config = SQLiteDatabaseConfiguration(
            dbFile.absolutePath,
            SQLiteDatabase.CREATE_IF_NECESSARY or SQLiteDatabase.OPEN_READWRITE
        )
        config.customExtensions.add(SQLiteCustomExtension(soFile.absolutePath, "sqlite3_vec_init"))
        db = SQLiteDatabase.openDatabase(config, null, null)

        db.execSQL(
            "CREATE TABLE IF NOT EXISTS chunk_meta (" +
                "id INTEGER PRIMARY KEY, patientId TEXT, documentType TEXT, documentId TEXT, " +
                "content TEXT, dateString TEXT, timestamp INTEGER)"
        )
        db.execSQL("CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[$dimensions])")
    }

    fun insertChunks(chunks: List<ClinicalChunk>) {
        db.beginTransaction()
        try {
            for (chunk in chunks) {
                val embedding = chunk.embedding
                if (embedding == null || embedding.size != dimensions) continue
                val values = ContentValues().apply {
                    put("id", chunk.id)
                    put("patientId", chunk.patientId)
                    put("documentType", chunk.documentType)
                    put("documentId", chunk.documentId)
                    put("content", chunk.content)
                    put("dateString", chunk.dateString)
                    put("timestamp", chunk.timestamp)
                }
                db.insert("chunk_meta", null, values)
                db.execSQL(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    arrayOf<Any>(chunk.id, serializeFloat32(embedding))
                )
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    fun searchTopK(queryEmbedding: FloatArray, topK: Int): List<ScoredChunk> {
        val cursor = db.rawQuery(
            "WITH knn AS (" +
                "SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? ORDER BY distance LIMIT $topK" +
                ") " +
                "SELECT m.id, m.patientId, m.documentType, m.documentId, m.content, m.dateString, " +
                "m.timestamp, knn.distance " +
                "FROM knn JOIN chunk_meta m ON m.id = knn.rowid " +
                "ORDER BY knn.distance",
            arrayOf<Any>(serializeFloat32(queryEmbedding))
        )
        val results = mutableListOf<ScoredChunk>()
        cursor.use {
            while (it.moveToNext()) {
                val chunk = ClinicalChunk(
                    id = it.getLong(0),
                    patientId = it.getString(1),
                    documentType = it.getString(2),
                    documentId = it.getString(3),
                    content = it.getString(4),
                    dateString = it.getString(5),
                    timestamp = it.getLong(6)
                )
                val distance = it.getDouble(7)
                results.add(ScoredChunk(chunk = chunk, score = (1.0 / (1.0 + distance) * 100).toFloat()))
            }
        }
        return results
    }

    fun diskSizeBytes(): Long = dbFile.length() +
        File(dbFile.parent, dbFile.name + "-wal").let { if (it.exists()) it.length() else 0L } +
        File(dbFile.parent, dbFile.name + "-shm").let { if (it.exists()) it.length() else 0L }

    fun close() {
        db.close()
        dbFile.delete()
    }

    companion object {
        private fun serializeFloat32(vector: FloatArray): ByteArray {
            val buffer = ByteBuffer.allocate(vector.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (f in vector) buffer.putFloat(f)
            return buffer.array()
        }
    }
}
