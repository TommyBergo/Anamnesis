/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo

import android.content.Context

/**
 * Represents a downloadable model with its associated files.
 */
data class ModelInfo(
    val displayName: String,
    val modelUrl: String,
    val modelFilename: String,
    val tokenizerUrl: String,
    val tokenizerFilename: String,
    val modelType: ModelType
) {
    fun hasTokenizer(): Boolean = tokenizerUrl.isNotEmpty()
}

/** Maps model display names to their download URLs, loaded from JSON configuration at runtime via PresetConfigManager. */
object ModelDownloadConfig {

    private var configManager: PresetConfigManager? = null
    private var cachedModels: Map<String, ModelInfo> = emptyMap()

    fun initialize(context: Context) {
        if (configManager == null) {
            configManager = PresetConfigManager(context.applicationContext)
            reloadModels()
        }
    }

    fun reloadModels() {
        cachedModels = configManager?.loadModels() ?: emptyMap()
    }

    fun updateModels(models: Map<String, ModelInfo>) {
        cachedModels = models
    }

    fun getConfigManager(): PresetConfigManager? = configManager

    fun getAvailableModels(): Map<String, ModelInfo> = cachedModels

    fun getDisplayNames(): Array<String> =
        cachedModels.values.map { it.displayName }.toTypedArray()

    fun getModelKeys(): Array<String> = cachedModels.keys.toTypedArray()

    fun getByDisplayName(displayName: String): ModelInfo? =
        cachedModels.values.find { it.displayName == displayName }

    fun getByKey(key: String): ModelInfo? = cachedModels[key]
}
