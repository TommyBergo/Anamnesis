/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo

/** Defines the media capabilities supported by a model. */
enum class MediaCapability {
    TEXT,
    IMAGE,
    AUDIO
}

/** Identifies a model's family from its filename to select the matching chat template and media capabilities. */
enum class ModelType(
    private val patterns: Array<String>,
    val mediaCapabilities: Set<MediaCapability>
) {
    GEMMA_3(
        arrayOf("gemma"),
        setOf(MediaCapability.TEXT, MediaCapability.IMAGE)
    ),
    LLAMA_3(
        arrayOf("llama"),
        setOf(MediaCapability.TEXT)
    ),
    LLAVA_1_5(
        arrayOf("llava"),
        setOf(MediaCapability.TEXT, MediaCapability.IMAGE)
    ),
    LLAMA_GUARD_3(
        arrayOf("llama_guard", "llama-guard", "llamaguard"),
        setOf(MediaCapability.TEXT)
    ),
    QWEN_3(
        arrayOf("qwen"),
        setOf(MediaCapability.TEXT)
    ),
    VOXTRAL(
        arrayOf("voxtral"),
        setOf(MediaCapability.TEXT, MediaCapability.AUDIO)
    ),
    PHI_4(
        arrayOf("phi_4", "phi-4", "phi4"),
        setOf(MediaCapability.TEXT)
    ),
    SMOLLM_3(
        arrayOf("smollm3", "smollm_3", "smollm"),
        setOf(MediaCapability.TEXT)
    ),
    LFM2(
        arrayOf("lfm2", "lfm"),
        setOf(MediaCapability.TEXT)
    );

    fun supportsImage(): Boolean = mediaCapabilities.contains(MediaCapability.IMAGE)

    fun supportsAudio(): Boolean = mediaCapabilities.contains(MediaCapability.AUDIO)

    fun isTextOnly(): Boolean = mediaCapabilities.size == 1 && mediaCapabilities.contains(MediaCapability.TEXT)

    private fun matchesFileName(lowerFileName: String): Boolean {
        return patterns.any { lowerFileName.contains(it) }
    }

    companion object {
        @JvmStatic
        fun fromFilePath(filePath: String?): ModelType? {
            if (filePath.isNullOrEmpty()) {
                return null
            }

            val fileName = filePath.substringAfterLast('/')
            val lowerFileName = fileName.lowercase()

            if (LLAMA_GUARD_3.matchesFileName(lowerFileName)) {
                return LLAMA_GUARD_3
            }
            if (LLAVA_1_5.matchesFileName(lowerFileName)) {
                return LLAVA_1_5
            }

            return values().firstOrNull { type ->
                type != LLAMA_GUARD_3 && type != LLAVA_1_5 && type.matchesFileName(lowerFileName)
            }
        }
    }
}
