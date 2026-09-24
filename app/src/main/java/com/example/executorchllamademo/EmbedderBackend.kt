/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo

import com.google.gson.annotations.SerializedName

/** Enumerates the three dense embedding backends selectable for clinical RAG retrieval, matching the paper's Table 3 rows. */
enum class EmbedderBackend(val displayName: String, val description: String) {
    @SerializedName("MEDIAPIPE_GEMMA")
    EMBEDDING_GEMMA("EmbeddingGemma", "EmbeddingGemma 300M via MediaPipe, 512-token window"),
    GRANITE("Granite", "IBM Granite Embedding 311M multilingual, 512-token window"),
    MULTILINGUAL_E5_SMALL("E5 multilingual small", "intfloat/multilingual-e5-small 118M, 384-dim, 512-token window");

    companion object {
        fun fromDisplayName(name: String): EmbedderBackend {
            return values().find { it.displayName == name } ?: EMBEDDING_GEMMA
        }
    }
}
