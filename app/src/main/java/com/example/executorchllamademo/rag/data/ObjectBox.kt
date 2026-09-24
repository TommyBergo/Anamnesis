package com.example.executorchllamademo.rag.data

import android.content.Context
import io.objectbox.BoxStore
import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.rag.data.MyObjectBox

/** Initializes and holds the singleton ObjectBox BoxStore used by the ObjectBox-backed vector repository. */
object ObjectBox {
    lateinit var store: BoxStore
        private set

    fun init(context: Context) {
        if (!::store.isInitialized) {
            store = MyObjectBox.builder()
                .androidContext(context.applicationContext)
                .build()
        }
    }
}