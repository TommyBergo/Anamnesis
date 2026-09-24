/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo

/** Holds module-specific settings for the current model/tokenizer configuration, supporting both legacy single-model and multi-model LoRA setups. */
data class ModuleSettings(
    val modelFilePath: String = "",
    val tokenizerFilePath: String = "",
    val dataPath: String = "",
    val temperature: Double = DEFAULT_TEMPERATURE,
    val systemPrompt: String = "",
    val userPrompt: String = PromptFormat.getUserPromptTemplate(DEFAULT_MODEL),
    val modelType: ModelType = DEFAULT_MODEL,
    val backendType: BackendType = DEFAULT_BACKEND,
    val isClearChatHistory: Boolean = false,
    val isLoadModel: Boolean = false,

    val isLoraMode: Boolean = false,

    val foundationDataPath: String = "",

    val models: List<ModelConfiguration> = emptyList(),
    val activeModelId: String = "",
    val sharedDataPath: String = "",
    val foundationModelType: ModelType = ModelType.LLAMA_3
) {
    fun getEffectiveModelType(): ModelType {
        val activeModel = getActiveModel()
        return activeModel?.modelType ?: modelType
    }

    fun getFormattedSystemPrompt(): String {
        return PromptFormat.getSystemPromptTemplate(getEffectiveModelType())
            .replace(PromptFormat.SYSTEM_PLACEHOLDER, systemPrompt)
    }

    fun getFormattedUserPrompt(prompt: String, thinkingMode: Boolean): String {
        val effectiveType = getEffectiveModelType()
        return userPrompt
            .replace(PromptFormat.USER_PLACEHOLDER, prompt)
            .replace(
                PromptFormat.THINKING_MODE_PLACEHOLDER,
                PromptFormat.getThinkingModeToken(effectiveType, thinkingMode)
            )
    }

    fun getActiveModel(): ModelConfiguration? {
        if (models.isEmpty() || activeModelId.isEmpty()) return null
        return models.find { it.id == activeModelId }
    }

    fun getModelById(modelId: String): ModelConfiguration? {
        return models.find { it.id == modelId }
    }

    fun getEffectiveDataPath(): String {
        return foundationDataPath.ifEmpty { sharedDataPath.ifEmpty { dataPath } }
    }

    fun hasMultipleModels(): Boolean = models.size > 1

    fun hasModels(): Boolean = models.isNotEmpty()

    fun addModel(model: ModelConfiguration): ModuleSettings {
        val existingIndex = models.indexOfFirst { it.id == model.id }
        val newModels = if (existingIndex >= 0) {
            models.toMutableList().apply { this[existingIndex] = model }
        } else {
            models + model
        }
        val newActiveId = if (models.isEmpty()) model.id else activeModelId
        return copy(models = newModels, activeModelId = newActiveId)
    }

    fun removeModel(modelId: String): ModuleSettings {
        val newModels = models.filter { it.id != modelId }
        val newActiveId = if (activeModelId == modelId) {
            newModels.firstOrNull()?.id ?: ""
        } else {
            activeModelId
        }
        return copy(models = newModels, activeModelId = newActiveId)
    }

    fun setActiveModel(modelId: String): ModuleSettings {
        return copy(activeModelId = modelId)
    }

    fun migrateToMultiModel(): ModuleSettings {
        if (models.isNotEmpty()) return this

        if (modelFilePath.isEmpty() || tokenizerFilePath.isEmpty()) return this

        val legacyModel = ModelConfiguration.create(
            modelFilePath = modelFilePath,
            tokenizerFilePath = tokenizerFilePath,
            modelType = modelType,
            backendType = backendType,
            temperature = temperature
        )

        return copy(
            models = listOf(legacyModel),
            activeModelId = legacyModel.id,
            sharedDataPath = dataPath
        )
    }

    companion object {
        const val DEFAULT_TEMPERATURE = 0.0
        val DEFAULT_MODEL = ModelType.LLAMA_3
        val DEFAULT_BACKEND = BackendType.XNNPACK
    }
}
