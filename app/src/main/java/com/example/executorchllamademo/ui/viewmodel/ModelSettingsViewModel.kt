/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo.ui.viewmodel

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import com.example.executorchllamademo.AppearanceMode
import com.example.executorchllamademo.AppSettings
import com.example.executorchllamademo.BackendType
import com.example.executorchllamademo.DemoSharedPreferences
import com.example.executorchllamademo.ModelConfiguration
import com.example.executorchllamademo.ModelSettingsActivity
import com.example.executorchllamademo.ModelType
import com.example.executorchllamademo.ModuleSettings
import com.example.executorchllamademo.PromptFormat

/** View model for the model-settings screen, managing single-model and multi-model (LoRA) configuration state and the add/remove-model flow. */
class ModelSettingsViewModel : ViewModel() {

    var moduleSettings by mutableStateOf(ModuleSettings())
        private set

    var appSettings by mutableStateOf(AppSettings())
        private set

    var showBackendDialog by mutableStateOf(false)
    var showModelDialog by mutableStateOf(false)
    var showTokenizerDialog by mutableStateOf(false)
    var showDataPathDialog by mutableStateOf(false)
    var showFoundationDataPathDialog by mutableStateOf(false)
    var showFoundationModelTypeDialog by mutableStateOf(false)
    var showAdapterDialog by mutableStateOf(false)
    var showModelTypeDialog by mutableStateOf(false)
    var showLoadModelDialog by mutableStateOf(false)
    var showResetSystemPromptDialog by mutableStateOf(false)
    var showResetUserPromptDialog by mutableStateOf(false)
    var showInvalidPromptDialog by mutableStateOf(false)
    var showAppearanceDialog by mutableStateOf(false)
    var showAddModelDialog by mutableStateOf(false)
    var showRemoveModelDialog by mutableStateOf(false)
    var showMemoryWarningDialog by mutableStateOf(false)

    var addModelStep by mutableStateOf(0)
        private set
    var tempModelPath by mutableStateOf("")
        private set
    var tempTokenizerPath by mutableStateOf("")
        private set
    var tempModelType by mutableStateOf(ModelType.LLAMA_3)
        private set
    var tempAdapterPaths by mutableStateOf<List<String>>(emptyList())
        private set

    var modelToRemove by mutableStateOf<String?>(null)
        private set

    var modelFiles by mutableStateOf<Array<String>>(emptyArray())
        private set
    var tokenizerFiles by mutableStateOf<Array<String>>(emptyArray())
        private set
    var dataPathFiles by mutableStateOf<Array<String>>(emptyArray())
        private set

    private var demoSharedPreferences: DemoSharedPreferences? = null

    fun initialize(context: Context) {
        demoSharedPreferences = DemoSharedPreferences(context)
        loadSettings()
        refreshFileLists()
    }

    private fun loadSettings() {
        demoSharedPreferences?.let { prefs ->
            var settings = prefs.getModuleSettings()
            if (settings.isLoraMode) {
                settings = settings.migrateToMultiModel()
            }
            moduleSettings = settings
            appSettings = prefs.getAppSettings()
        }
    }

    fun saveSettings() {
        demoSharedPreferences?.saveModuleSettings(moduleSettings)
        demoSharedPreferences?.saveAppSettings(appSettings)
    }

    fun refreshFileLists() {
        modelFiles = ModelSettingsActivity.listLocalFile("/data/local/tmp/llama/", arrayOf(".pte"))
        tokenizerFiles = ModelSettingsActivity.listLocalFile("/data/local/tmp/llama/", arrayOf(".bin", ".json", ".model"))
        dataPathFiles = ModelSettingsActivity.listLocalFile("/data/local/tmp/llama/", arrayOf(".ptd"))
    }

    fun selectBackend(backendType: BackendType) {
        var newSettings = moduleSettings.copy(backendType = backendType)
        newSettings = applyBackendDefaults(newSettings)
        moduleSettings = newSettings
    }

    private fun applyBackendDefaults(settings: ModuleSettings): ModuleSettings {
        return if (settings.backendType == BackendType.MEDIATEK) {
            settings.copy(
                modelFilePath = settings.modelFilePath.ifEmpty { "/in/mtk/llama/runner" },
                tokenizerFilePath = settings.tokenizerFilePath.ifEmpty { "/in/mtk/llama/runner" }
            )
        } else {
            settings
        }
    }

    fun selectModel(modelPath: String) {
        var newSettings = moduleSettings.copy(modelFilePath = modelPath)
        newSettings = autoSelectModelType(newSettings, modelPath)
        moduleSettings = newSettings
    }

    private fun autoSelectModelType(settings: ModuleSettings, filePath: String): ModuleSettings {
        val detectedType = ModelType.fromFilePath(filePath)
        return if (detectedType != null) {
            settings.copy(
                modelType = detectedType,
                userPrompt = PromptFormat.getUserPromptTemplate(detectedType)
            )
        } else {
            settings
        }
    }

    fun selectTokenizer(tokenizerPath: String) {
        moduleSettings = moduleSettings.copy(tokenizerFilePath = tokenizerPath)
    }

    fun selectDataPath(dataPath: String) {
        moduleSettings = moduleSettings.copy(
            dataPath = dataPath,
            sharedDataPath = dataPath
        )
    }

    fun selectFoundationDataPath(dataPath: String) {
        moduleSettings = moduleSettings.copy(foundationDataPath = dataPath)
    }

    fun selectFoundationModelType(modelType: ModelType) {
        moduleSettings = moduleSettings.copy(foundationModelType = modelType)
        showFoundationModelTypeDialog = false
    }

    fun selectModelType(modelType: ModelType) {
        moduleSettings = moduleSettings.copy(
            modelType = modelType,
            userPrompt = PromptFormat.getUserPromptTemplate(modelType)
        )
    }

    fun updateTemperature(temperature: Double) {
        moduleSettings = moduleSettings.copy(
            temperature = temperature,
            isLoadModel = true
        )
        saveSettings()
    }

    fun updateSystemPrompt(prompt: String) {
        moduleSettings = moduleSettings.copy(systemPrompt = prompt)
    }

    fun resetSystemPrompt() {
        moduleSettings = moduleSettings.copy(systemPrompt = PromptFormat.DEFAULT_SYSTEM_PROMPT)
    }

    fun updateUserPrompt(prompt: String) {
        if (isValidUserPrompt(prompt)) {
            moduleSettings = moduleSettings.copy(userPrompt = prompt)
        } else {
            showInvalidPromptDialog = true
        }
    }

    fun resetUserPrompt() {
        moduleSettings = moduleSettings.copy(
            userPrompt = PromptFormat.getUserPromptTemplate(moduleSettings.modelType)
        )
    }

    private fun isValidUserPrompt(userPrompt: String): Boolean {
        return userPrompt.contains(PromptFormat.USER_PLACEHOLDER)
    }

    fun confirmLoadModel() {
        saveSettings()
        moduleSettings = moduleSettings.copy(isLoadModel = true)
    }

    fun confirmClearChat() {
        moduleSettings = moduleSettings.copy(isClearChatHistory = true)
        saveSettings()
    }

    fun isLoadModelEnabled(): Boolean {
        if (moduleSettings.hasModels()) {
            return moduleSettings.models.any { it.isValid() }
        }
        return moduleSettings.modelFilePath.isNotEmpty() && moduleSettings.tokenizerFilePath.isNotEmpty()
    }

    fun isMediaTekMode(): Boolean {
        return moduleSettings.backendType == BackendType.MEDIATEK
    }

    fun getFilenameFromPath(path: String): String {
        return if (path.isEmpty()) "" else path.substringAfterLast('/')
    }

    fun selectAppearanceMode(mode: AppearanceMode) {
        appSettings = appSettings.copy(appearanceMode = mode)
        demoSharedPreferences?.saveAppSettings(appSettings)
    }

    fun toggleLoraMode(enabled: Boolean) {
        moduleSettings = moduleSettings.copy(isLoraMode = enabled)
    }

    fun startAddModel() {
        tempModelPath = ""
        tempTokenizerPath = ""
        tempModelType = ModelType.LLAMA_3
        tempAdapterPaths = emptyList()
        addModelStep = 1
        showAddModelDialog = true
        refreshFileLists()
    }

    fun selectTempModel(modelPath: String) {
        tempModelPath = modelPath
        val detectedType = ModelType.fromFilePath(modelPath)
        if (detectedType != null) {
            tempModelType = detectedType
        }
        addModelStep = 2
    }

    fun selectTempTokenizer(tokenizerPath: String) {
        tempTokenizerPath = tokenizerPath
        tempModelType = moduleSettings.foundationModelType
        addModelStep = 3
    }

    fun selectTempModelType(modelType: ModelType) {
        tempModelType = modelType
    }

    fun goToAddModelStep(step: Int) {
        addModelStep = step
    }

    fun confirmAddModel() {
        if (tempModelPath.isEmpty() || tempTokenizerPath.isEmpty()) return

        val newModel = ModelConfiguration.create(
            modelFilePath = tempModelPath,
            tokenizerFilePath = tempTokenizerPath,
            modelType = tempModelType,
            backendType = moduleSettings.backendType,
            temperature = ModuleSettings.DEFAULT_TEMPERATURE
        ).copy(adapterFilePaths = tempAdapterPaths)

        moduleSettings = moduleSettings.addModel(newModel)
        cancelAddModel()
    }

    fun cancelAddModel() {
        showAddModelDialog = false
        addModelStep = 0
        tempModelPath = ""
        tempTokenizerPath = ""
        tempModelType = ModelType.LLAMA_3
        tempAdapterPaths = emptyList()
    }

    fun previousAddModelStep() {
        when (addModelStep) {
            2 -> {
                addModelStep = 1
                tempTokenizerPath = ""
            }
            3 -> {
                addModelStep = 2
            }
            else -> cancelAddModel()
        }
    }

    fun selectActiveModel(modelId: String) {
        moduleSettings = moduleSettings.setActiveModel(modelId)
    }

    fun initiateRemoveModel(modelId: String) {
        modelToRemove = modelId
        showRemoveModelDialog = true
    }

    fun confirmRemoveModel() {
        modelToRemove?.let { modelId ->
            moduleSettings = moduleSettings.removeModel(modelId)
        }
        cancelRemoveModel()
    }

    fun cancelRemoveModel() {
        showRemoveModelDialog = false
        modelToRemove = null
    }

    fun shouldShowMemoryWarning(): Boolean {
        return moduleSettings.models.size > 2
    }

    fun initiateLoadModels() {
        if (shouldShowMemoryWarning()) {
            showMemoryWarningDialog = true
        } else {
            showLoadModelDialog = true
        }
    }

    fun proceedAfterMemoryWarning() {
        showMemoryWarningDialog = false
        showLoadModelDialog = true
    }

    fun addTempAdapter(adapterPath: String) {
        if (adapterPath.isNotEmpty() && !tempAdapterPaths.contains(adapterPath)) {
            tempAdapterPaths = tempAdapterPaths + adapterPath
        }
    }

    fun removeTempAdapter(adapterPath: String) {
        tempAdapterPaths = tempAdapterPaths.filter { it != adapterPath }
    }
}
