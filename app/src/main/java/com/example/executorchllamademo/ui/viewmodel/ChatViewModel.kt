/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo.ui.viewmodel

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import com.example.executorchllamademo.BackendType
import com.example.executorchllamademo.AppSettings
import com.example.executorchllamademo.DemoSharedPreferences
import com.example.executorchllamademo.ETImage
import com.example.executorchllamademo.ETLogging
import com.example.executorchllamademo.Message
import com.example.executorchllamademo.MessageType
import com.example.executorchllamademo.ModelConfiguration
import com.example.executorchllamademo.ModelType
import com.example.executorchllamademo.ModelUtils
import com.example.executorchllamademo.PromptFormat
import com.example.executorchllamademo.ModuleSettings
import com.example.executorchllamademo.rag.data.ClinicalChunk
import com.example.executorchllamademo.EmbedderBackend
import com.example.executorchllamademo.rag.embedding.Embedder
import com.example.executorchllamademo.rag.embedding.GraniteEmbedder
import com.example.executorchllamademo.rag.embedding.MultilingualE5SmallEmbedder
import com.example.executorchllamademo.rag.embedding.EmbeddingGemmaEmbedder
import com.example.executorchllamademo.rag.embedding.PureEmbedder
import com.example.executorchllamademo.rag.embedding.embeddingFingerprint
import com.example.executorchllamademo.rag.repository.HybridBm25MediaPipeRepository
import com.example.executorchllamademo.rag.data.ObjectBox
import com.example.executorchllamademo.rag.RrfBm25MediaPipeRagEngine
import com.example.executorchllamademo.rag.MedicalPromptBuilder
import com.example.executorchllamademo.rag.ClinicalPdfProcessor
import kotlinx.coroutines.runBlocking
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.InstanceCreator
import com.google.gson.reflect.TypeToken
import org.json.JSONException
import org.json.JSONObject
import org.pytorch.executorch.ExecutorchRuntimeException
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule
import kotlinx.coroutines.flow.asStateFlow
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Paths
import java.util.concurrent.Executor
import java.util.concurrent.Executors

/** Represents the lifecycle state of the RAG engine's initialization: idle, loading, ready, or failed with an error message. */
sealed interface RagState {
    object Idle : RagState
    object Loading : RagState
    object Ready : RagState
    data class Error(val message: String) : RagState
}

/** View model driving the chat screen: model loading (including LoRA multi-model), message generation via [LlmModule], and RAG-based prompt assembly. */
class ChatViewModel(application: Application) : AndroidViewModel(application), LlmCallback {

    var inputText by mutableStateOf("")
    var isModelReady by mutableStateOf(false)
    var isGenerating by mutableStateOf(false)
    var showMediaSelector by mutableStateOf(false)
    var ramUsage by mutableStateOf("0 MB")
    var showMediaButtons by mutableStateOf(false)
    var supportsImageInput by mutableStateOf(false)
    var supportsAudioInput by mutableStateOf(false)

    private var isInThinkingBlock = false

    private val _selectedImages = mutableStateListOf<Uri>()
    val selectedImages: List<Uri> = _selectedImages

    var showModelLoadErrorDialog by mutableStateOf(false)
    var modelLoadError by mutableStateOf("")

    var isLoraMode by mutableStateOf(false)
        private set
    var availableModels by mutableStateOf<List<ModelConfiguration>>(emptyList())
        private set
    var activeModelId by mutableStateOf("")
        private set

    private val loadedModules = mutableMapOf<String, LlmModule>()

    private var module: LlmModule? = null
    private var resultMessage: Message? = null
    private val demoSharedPreferences = DemoSharedPreferences(application)
    private var currentSettingsFields = ModuleSettings()
    private var appSettings = AppSettings()
    private var promptID = 0
    private var sawStartHeaderId = false
    private var audioFileToPrefill: String? = null
    private var shouldAddSystemPrompt = true

    private val executor: Executor = Executors.newSingleThreadExecutor()
    private val contentResolver = application.contentResolver

    private var currentPatientId: String = "default_patient"
    private val pdfProcessor = ClinicalPdfProcessor(getApplication())
    private var embedder: Embedder = PureEmbedder()
    private lateinit var vectorRepository: HybridBm25MediaPipeRepository

    private val _ragState = MutableStateFlow<RagState>(RagState.Idle)
    val ragState: StateFlow<RagState> = _ragState.asStateFlow()
    val _messages = mutableStateListOf<Message>()
    val messages: List<Message> get() = _messages

    var ragEngine: RrfBm25MediaPipeRagEngine? = null
        private set

    init {
        viewModelScope.launch(Dispatchers.IO) {
            initRagEngine()
        }

        val moduleSettings = demoSharedPreferences.getModuleSettings()
        appSettings = demoSharedPreferences.getAppSettings()

        if (moduleSettings.isClearChatHistory) {
            demoSharedPreferences.removeExistingMessages()
            demoSharedPreferences.saveModuleSettings(moduleSettings.copy(isClearChatHistory = false))
        } else {
            loadSavedMessages()
        }
    }

    private suspend fun initRagEngine() {
        _ragState.value = RagState.Loading
        try {
            ObjectBox.init(getApplication())
            val boxStore = ObjectBox.store

            vectorRepository = HybridBm25MediaPipeRepository(boxStore)

            val selectedBackend = demoSharedPreferences.getAppSettings().embedderBackend
            Log.i(RAG_TAG, "[INIT] constructing embedder for selected backend=$selectedBackend")

            embedder = try {
                when (selectedBackend) {
                    EmbedderBackend.EMBEDDING_GEMMA -> EmbeddingGemmaEmbedder(getApplication())
                    EmbedderBackend.GRANITE -> GraniteEmbedder(getApplication())
                    EmbedderBackend.MULTILINGUAL_E5_SMALL -> MultilingualE5SmallEmbedder(getApplication())
                }
            } catch (e: Exception) {
                Log.e(RAG_TAG, "$selectedBackend embedder unavailable, falling back to PureEmbedder: ${e.message}", e)
                PureEmbedder()
            }

            Log.i(RAG_TAG, "[INIT] active embedder='${embedder.backendId}' dim=${embedder.embeddingDimension}")

            ragEngine = RrfBm25MediaPipeRagEngine(vectorRepository, embedder)

            _ragState.value = if (embedder !is PureEmbedder) {
                RagState.Ready
            } else {
                RagState.Error(
                    "Semantic embedder unavailable; using degraded fallback '${embedder.backendId}'. " +
                        "Clinical retrieval is disabled until the selected embedding model loads successfully."
                )
            }

            Log.d(RAG_TAG, "RAG Engine initialized successfully!")
        } catch (e: Exception) {
            Log.e(RAG_TAG, "Error during RAG Engine initialization: ${e.message}", e)
            _ragState.value = RagState.Error(e.message ?: "Unknown RAG initialization error")
        }
    }

    private fun loadSavedMessages() {
        val appSettings = demoSharedPreferences.getAppSettings()
        if (!appSettings.saveChatHistory) {
            demoSharedPreferences.removeExistingMessages()
            return
        }

        val existingMsgJSON = demoSharedPreferences.getSavedMessages()
        if (existingMsgJSON.isNotEmpty()) {
            val gson = GsonBuilder()
                .registerTypeAdapter(Message::class.java, InstanceCreator<Message> {
                    Message("", false, MessageType.TEXT, 0)
                })
                .create()
            val type = object : TypeToken<ArrayList<Message>>() {}.type
            val savedMessages: ArrayList<Message>? = gson.fromJson(existingMsgJSON, type)
            savedMessages?.let {
                _messages.addAll(it)
                promptID = _messages.maxOfOrNull { msg -> msg.promptID }?.plus(1) ?: 0
            }
        }
    }

    fun saveMessages() {
        val appSettings = demoSharedPreferences.getAppSettings()
        if (appSettings.saveChatHistory) {
            demoSharedPreferences.addMessages(_messages.toList())
        } else {
            demoSharedPreferences.removeExistingMessages()
        }
    }

    private val systemPromptMessage = "To get started, select your desired model and tokenizer from the top right corner"

    fun checkAndLoadSettings() {
        val updatedSettingsFields = demoSharedPreferences.getModuleSettings()
        appSettings = demoSharedPreferences.getAppSettings()
        val isUpdated = currentSettingsFields != updatedSettingsFields
        val isLoadModel = updatedSettingsFields.isLoadModel

        isLoraMode = updatedSettingsFields.isLoraMode
        availableModels = updatedSettingsFields.models
        activeModelId = updatedSettingsFields.activeModelId

        if (isUpdated) {
            val settingsAfterClear = checkForClearChatHistory(updatedSettingsFields)

            if (isLoadModel) {
                val settingsWithLoadFlagCleared = settingsAfterClear.copy(isLoadModel = false)
                currentSettingsFields = settingsWithLoadFlagCleared
                demoSharedPreferences.saveModuleSettings(settingsWithLoadFlagCleared)

                setBackendMode(settingsAfterClear.backendType)

                if (isLoraMode && settingsAfterClear.hasModels()) {
                    loadLoraModels(settingsAfterClear)
                } else {
                    loadLocalModelAndParameters(
                        settingsAfterClear.modelFilePath,
                        settingsAfterClear.tokenizerFilePath,
                        settingsAfterClear.dataPath,
                        settingsAfterClear.temperature.toFloat()
                    )
                }
            } else {
                currentSettingsFields = settingsAfterClear.copy()
                setBackendMode(settingsAfterClear.backendType)
                if (module == null && loadedModules.isEmpty()) {
                    addSystemMessage(systemPromptMessage)
                }
            }
        } else {
            setBackendMode(updatedSettingsFields.backendType)
            val modelPath = updatedSettingsFields.modelFilePath
            val tokenizerPath = updatedSettingsFields.tokenizerFilePath
            if (modelPath.isEmpty() || tokenizerPath.isEmpty()) {
                if (!isLoraMode || !updatedSettingsFields.hasModels()) {
                    addSystemMessage(systemPromptMessage)
                }
            }
        }
    }

    private fun loadLoraModels(settings: ModuleSettings) {
        Thread {
            val sharedDataPath = settings.getEffectiveDataPath()

            val loadingDetails = StringBuilder("Loading ${settings.models.size} LoRA model(s):\n")
            settings.models.forEachIndexed { index, modelConfig ->
                if (modelConfig.isValid()) {
                    val dataFiles = mutableListOf<String>()
                    if (sharedDataPath.isNotEmpty()) {
                        dataFiles.add(sharedDataPath)
                    }
                    dataFiles.addAll(modelConfig.adapterFilePaths)

                    loadingDetails.append("\n${index + 1}. ${modelConfig.displayName}\n")
                    loadingDetails.append("   PTE: ${modelConfig.modelFilePath.substringAfterLast('/')}\n")
                    loadingDetails.append("   Tokenizer: ${modelConfig.tokenizerFilePath.substringAfterLast('/')}\n")
                    loadingDetails.append("   Data files: ${if (dataFiles.isEmpty()) "none" else dataFiles.joinToString(", ") { it.substringAfterLast('/') }}\n")
                }
            }

            val modelLoadingMessage = Message(loadingDetails.toString(), false, MessageType.SYSTEM, 0)
            _messages.add(modelLoadingMessage)
            isModelReady = false

            var loadedCount = 0
            var firstLoadedModelId: String? = null

            for (modelConfig in settings.models) {
                if (!modelConfig.isValid()) continue

                try {
                    val dataFiles = mutableListOf<String>()
                    if (sharedDataPath.isNotEmpty()) {
                        dataFiles.add(sharedDataPath)
                    }
                    dataFiles.addAll(modelConfig.adapterFilePaths)

                    val dataFilesLog = if (dataFiles.isEmpty()) "no data files" else dataFiles.joinToString(", ")
                    ETLogging.getInstance().log(
                        "LoRA: Loading model ${modelConfig.displayName} with tokenizer ${modelConfig.tokenizerFilePath}, data files: $dataFilesLog"
                    )

                    val runStartTime = System.currentTimeMillis()
                    val llmModule = LlmModule(
                        ModelUtils.getModelCategory(modelConfig.modelType, modelConfig.backendType),
                        modelConfig.modelFilePath,
                        modelConfig.tokenizerFilePath,
                        modelConfig.temperature.toFloat(),
                        dataFiles
                    )

                    llmModule.load()
                    val loadDuration = System.currentTimeMillis() - runStartTime

                    loadedModules[modelConfig.id] = llmModule
                    loadedCount++

                    if (firstLoadedModelId == null) {
                        firstLoadedModelId = modelConfig.id
                    }

                    ETLogging.getInstance().log(
                        "LoRA: Loaded ${modelConfig.displayName} in ${loadDuration.toFloat() / 1000} sec"
                    )
                } catch (e: ExecutorchRuntimeException) {
                    ETLogging.getInstance().log("LoRA: Failed to load ${modelConfig.displayName}: ${e.message}")
                    _messages.add(Message("Failed to load ${modelConfig.displayName}: ${e.message}", false, MessageType.SYSTEM, 0))
                }
            }

            _messages.remove(modelLoadingMessage)

            if (loadedCount > 0) {
                val activeId = if (settings.activeModelId.isNotEmpty() && loadedModules.containsKey(settings.activeModelId)) {
                    settings.activeModelId
                } else {
                    firstLoadedModelId ?: ""
                }

                activeModelId = activeId
                module = loadedModules[activeId]

                val activeModelName = settings.getModelById(activeId)?.displayName ?: "Unknown"
                _messages.add(Message(
                    "Successfully loaded $loadedCount model(s). Active: $activeModelName. Use the switch button to change models.",
                    false, MessageType.SYSTEM, 0
                ))
                isModelReady = true
            } else {
                _messages.add(Message("No models loaded. Please check your configuration.", false, MessageType.SYSTEM, 0))
                isModelReady = false
            }
        }.start()
    }

    fun switchToModel(modelId: String) {
        if (!isLoraMode) return
        if (isGenerating) {
            addSystemMessage("Cannot switch models while generating. Please wait or stop generation.")
            return
        }

        val modelConfig = currentSettingsFields.getModelById(modelId)
        if (modelConfig == null) {
            addSystemMessage("Model not found.")
            return
        }

        if (loadedModules.containsKey(modelId)) {
            module = loadedModules[modelId]
            activeModelId = modelId

            currentSettingsFields = currentSettingsFields.setActiveModel(modelId)
            demoSharedPreferences.saveModuleSettings(currentSettingsFields)

            addSystemMessage("Switched to ${modelConfig.displayName}")
            ETLogging.getInstance().log("LoRA: Switched to already loaded model ${modelConfig.displayName}")
        } else {
            Thread {
                val sharedDataPath = currentSettingsFields.getEffectiveDataPath()
                addSystemMessage("Loading ${modelConfig.displayName}...")
                isModelReady = false

                try {
                    val dataFiles = mutableListOf<String>()
                    if (sharedDataPath.isNotEmpty()) {
                        dataFiles.add(sharedDataPath)
                    }
                    dataFiles.addAll(modelConfig.adapterFilePaths)

                    val runStartTime = System.currentTimeMillis()
                    val llmModule = LlmModule(
                        ModelUtils.getModelCategory(modelConfig.modelType, modelConfig.backendType),
                        modelConfig.modelFilePath,
                        modelConfig.tokenizerFilePath,
                        modelConfig.temperature.toFloat(),
                        dataFiles
                    )

                    llmModule.load()
                    val loadDuration = System.currentTimeMillis() - runStartTime

                    loadedModules[modelId] = llmModule
                    module = llmModule
                    activeModelId = modelId

                    currentSettingsFields = currentSettingsFields.setActiveModel(modelId)
                    demoSharedPreferences.saveModuleSettings(currentSettingsFields)

                    addSystemMessage("Switched to ${modelConfig.displayName} (loaded in ${loadDuration.toFloat() / 1000} sec)")
                    ETLogging.getInstance().log("LoRA: Loaded and switched to ${modelConfig.displayName} in ${loadDuration.toFloat() / 1000} sec")
                    isModelReady = true
                } catch (e: ExecutorchRuntimeException) {
                    addSystemMessage("Failed to load ${modelConfig.displayName}: ${e.message}")
                    ETLogging.getInstance().log("LoRA: Failed to load ${modelConfig.displayName}: ${e.message}")
                    isModelReady = loadedModules.isNotEmpty()
                }
            }.start()
        }
    }

    private fun setBackendMode(backendType: BackendType) {
        val backendSupportsMedia = when (backendType) {
            BackendType.XNNPACK, BackendType.QUALCOMM, BackendType.VULKAN -> true
            BackendType.MEDIATEK -> false
        }
        updateMediaCapabilities(backendSupportsMedia)
    }

    private fun updateMediaCapabilities(backendSupportsMedia: Boolean) {
        val modelType = if (currentSettingsFields.isLoraMode) {
            currentSettingsFields.foundationModelType
        } else {
            currentSettingsFields.modelType
        }

        supportsImageInput = backendSupportsMedia && modelType.supportsImage()
        supportsAudioInput = backendSupportsMedia && modelType.supportsAudio()
        showMediaButtons = supportsImageInput || supportsAudioInput
        ETLogging.getInstance().log("updateMediaCapabilities: modelType=$modelType, supportsImage=${modelType.supportsImage()}, supportsAudio=${modelType.supportsAudio()}, showMediaButtons=$showMediaButtons")
    }

    private fun getCapabilityDescription(modelType: ModelType): String {
        return when {
            modelType.supportsImage() && modelType.supportsAudio() ->
                "You can send text, images, or audio for inference."
            modelType.supportsImage() ->
                "You can send text or images for inference."
            modelType.supportsAudio() ->
                "You can send text or audio for inference."
            else ->
                "You can send text for inference."
        }
    }

    private fun checkForClearChatHistory(updatedSettingsFields: ModuleSettings): ModuleSettings {
        if (updatedSettingsFields.isClearChatHistory) {
            _messages.clear()
            demoSharedPreferences.removeExistingMessages()
            val clearedSettings = updatedSettingsFields.copy(isClearChatHistory = false)
            demoSharedPreferences.saveModuleSettings(clearedSettings)
            module?.resetContext()
            shouldAddSystemPrompt = true
            promptID = 0
            return clearedSettings
        }
        return updatedSettingsFields
    }

    private fun loadLocalModelAndParameters(
        modelFilePath: String,
        tokenizerFilePath: String,
        dataPath: String,
        temperature: Float
    ) {
        Thread {
            setLocalModel(modelFilePath, tokenizerFilePath, dataPath, temperature)
        }.start()
    }

    private fun setLocalModel(
        modelPath: String,
        tokenizerPath: String,
        dataPath: String,
        temperature: Float
    ) {
        val modelLoadingMessage = Message("Loading model...", false, MessageType.SYSTEM, 0)
        ETLogging.getInstance().log(
            "Loading model $modelPath with tokenizer $tokenizerPath data path $dataPath"
        )

        isModelReady = false
        _messages.add(modelLoadingMessage)

        val runStartTime = System.currentTimeMillis()
        module = if (dataPath.isEmpty()) {
            LlmModule(
                ModelUtils.getModelCategory(
                    currentSettingsFields.modelType,
                    currentSettingsFields.backendType
                ),
                modelPath,
                tokenizerPath,
                temperature
            )
        } else {
            LlmModule(
                ModelUtils.getModelCategory(
                    currentSettingsFields.modelType,
                    currentSettingsFields.backendType
                ),
                modelPath,
                tokenizerPath,
                temperature,
                dataPath
            )
        }

        var loadDuration = System.currentTimeMillis() - runStartTime
        var modelInfo: String

        var loadSuccess = false
        try {
            module?.load()
            val pteName = modelPath.substringAfterLast('/')
            val tokenizerName = tokenizerPath.substringAfterLast('/')
            val capabilityText = getCapabilityDescription(currentSettingsFields.modelType)
            modelInfo = "Successfully loaded model. $pteName and tokenizer $tokenizerName in ${loadDuration.toFloat() / 1000} sec. $capabilityText"

            if (currentSettingsFields.modelType == ModelType.LLAVA_1_5) {
                val llavaPresetPrompt = PromptFormat.getLlavaPresetPrompt()
                ETLogging.getInstance().log("Llava start prefill prompt: $llavaPresetPrompt")
                module?.prefillPrompt(llavaPresetPrompt)
                ETLogging.getInstance().log("Llava completes prefill prompt")
            }
            loadSuccess = true
        } catch (e: ExecutorchRuntimeException) {
            modelInfo = "Model load failure: ${e.message}"
            loadDuration = 0
            modelLoadError = modelInfo
            showModelLoadErrorDialog = true
        }

        val modelLoadedMessage = Message(modelInfo, false, MessageType.SYSTEM, 0)
        val modelLoggingInfo = "Model path: $modelPath\n" +
                "Tokenizer path: $tokenizerPath\n" +
                "Backend: ${currentSettingsFields.backendType}\n" +
                "ModelType: ${ModelUtils.getModelCategory(currentSettingsFields.modelType, currentSettingsFields.backendType)}\n" +
                "Temperature: $temperature\n" +
                "Model loaded time: $loadDuration ms"
        ETLogging.getInstance().log("Load complete. $modelLoggingInfo")

        isModelReady = loadSuccess
        _messages.remove(modelLoadingMessage)
        _messages.add(modelLoadedMessage)
    }

    fun updateMemoryUsage() {
        val context = getApplication<Application>()
        val memoryInfo = ActivityManager.MemoryInfo()
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            ?: return
        activityManager.getMemoryInfo(memoryInfo)
        val totalMem = memoryInfo.totalMem / (1024 * 1024)
        val availableMem = memoryInfo.availMem / (1024 * 1024)
        val usedMem = totalMem - availableMem
        ramUsage = "${usedMem}MB"
    }

    fun toggleMediaSelector() {
        showMediaSelector = !showMediaSelector
    }

    fun addImage(uri: Uri) {
        if (_selectedImages.size < MAX_NUM_OF_IMAGES) {
            _selectedImages.add(uri)
            prefillImageIfNeeded()
        }
    }

    fun removeImage(uri: Uri) {
        _selectedImages.remove(uri)
    }

    fun clearImages() {
        _selectedImages.clear()
    }

    fun setAudioFile(path: String) {
        audioFileToPrefill = path
        _messages.add(Message("Selected audio: $path", false, MessageType.SYSTEM, 0))
    }

    private fun prefillImageIfNeeded() {
        if (currentSettingsFields.modelType == ModelType.LLAVA_1_5 ||
            currentSettingsFields.modelType == ModelType.GEMMA_3
        ) {
            val processedImageList = getProcessedImagesForModel(_selectedImages)
            if (processedImageList.isNotEmpty()) {
                _messages.add(
                    Message("Starting image prefill.", false, MessageType.SYSTEM, 0)
                )
                executor.execute {
                    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_MORE_FAVORABLE)
                    ETLogging.getInstance().log("Starting runnable prefill image")
                    val img = processedImageList[0]
                    ETLogging.getInstance().log("Starting prefill image")
                    if (currentSettingsFields.modelType == ModelType.LLAVA_1_5) {
                        module?.prefillImages(
                            img.getInts(),
                            img.width,
                            img.height,
                            ModelUtils.VISION_MODEL_IMAGE_CHANNELS
                        )
                    } else if (currentSettingsFields.modelType == ModelType.GEMMA_3) {
                        val gemmaPreImagePrompt = PromptFormat.getGemmaPreImagePrompt()
                        ETLogging.getInstance().log("Gemma prefill pre-image prompt: $gemmaPreImagePrompt")
                        module?.prefillPrompt(gemmaPreImagePrompt)
                        module?.prefillImages(
                            img.getFloats(),
                            img.width,
                            img.height,
                            ModelUtils.VISION_MODEL_IMAGE_CHANNELS
                        )
                    }
                }
            }
        }
    }

    private fun getInputImageSideSize(): Int {
        return when (currentSettingsFields.modelType) {
            ModelType.LLAVA_1_5 -> 336
            ModelType.GEMMA_3 -> 896
            else -> throw IllegalArgumentException("Unsupported model type: ${currentSettingsFields.modelType}")
        }
    }

    private fun getProcessedImagesForModel(uris: List<Uri>): List<ETImage> {
        val imageList = mutableListOf<ETImage>()
        uris.forEach { uri ->
            imageList.add(ETImage(contentResolver, uri, getInputImageSideSize()))
        }
        return imageList
    }

    fun sendMessage() {
        if (inputText.trim().isEmpty()) return
        if (!isModelReady || isGenerating) return

        val rawPrompt = inputText

        val recentHistory = _messages
            .filter { it.messageType == MessageType.TEXT && it.text.isNotBlank() }
            .takeLast(4)
            .joinToString("\n") { msg ->
                val sender = if (msg.isSent) "User" else "Assistant"
                "$sender: ${msg.text}"
            }

        _messages.add(Message(rawPrompt, true, MessageType.TEXT, promptID))
        inputText = ""

        resultMessage = Message("", false, MessageType.TEXT, promptID)
        isInThinkingBlock = false
        _messages.add(resultMessage!!)

        promptID++

        executor.execute {
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_MORE_FAVORABLE)
            ETLogging.getInstance().log("starting runnable generate()")
            isGenerating = true

            val generateStartTime = System.currentTimeMillis()

            val safeRagEngine = ragEngine
            val finalPrompt: String = if (safeRagEngine != null) {
                try {
                    val ragOutput = runBlocking {
                        safeRagEngine.preparePrompt(
                            userQuery = rawPrompt,
                            patientId = currentPatientId,
                            patientBackgroundSummary = "Patient under monitoring for depressed mood.",
                            recentHistory = recentHistory,
                            topK = 2
                        )
                    }

                    resultMessage?.ragDebugHeader = ragOutput.chunkHeaderMarkdown
                    ragOutput.prompt

                } catch (e: Exception) {
                    Log.e(RAG_TAG, "[QUERY] RAG error, falling back to standard prompt: ${e.message}", e)
                    ETLogging.getInstance().log("RAG Engine error: ${e.localizedMessage}. Fallback to standard prompt.")
                    buildStandardPrompt(rawPrompt)
                }
            } else {
                ETLogging.getInstance().log("RAG Engine not initialized. Fallback to standard prompt.")
                buildStandardPrompt(rawPrompt)
            }

            module?.resetContext()

            if (currentSettingsFields.modelType == ModelType.LLAMA_GUARD_3) {
                val llamaGuardPromptForClassification =
                    PromptFormat.getFormattedLlamaGuardPrompt(rawPrompt)
                ETLogging.getInstance().log("Running inference.. prompt=$llamaGuardPromptForClassification")
                module?.generate(
                    llamaGuardPromptForClassification,
                    llamaGuardPromptForClassification.length + 64,
                    this,
                    false
                )
            } else {
                ETLogging.getInstance().log("Running inference with RAG prompt.. prompt=$finalPrompt")
                module?.generate(finalPrompt, appSettings.maxSeqLen, this, false)
            }

            val generateDuration = System.currentTimeMillis() - generateStartTime
            resultMessage?.let { msg ->
                msg.totalGenerationTime = generateDuration
                val index = _messages.indexOfLast { it === msg }
                if (index >= 0) {
                    val updated = msg.copy()
                    _messages[index] = updated
                    resultMessage = updated
                }
            }

            isGenerating = false
            ETLogging.getInstance().log("Inference completed")
        }
    }

    private fun buildStandardPrompt(rawPrompt: String): String {
        val systemPrompt = if (shouldAddSystemPrompt) currentSettingsFields.getFormattedSystemPrompt() else ""
        val userPrompt = currentSettingsFields.getFormattedUserPrompt(rawPrompt, thinkingMode = false)
        return systemPrompt + userPrompt
    }

    private fun deriveDocumentScopeId(uri: Uri): String {
        var displayName: String? = null
        try {
            contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (nameIndex >= 0) displayName = cursor.getString(nameIndex)
                }
            }
        } catch (e: Exception) {
            Log.w(RAG_TAG, "Could not resolve display name for $uri: ${e.message}")
        }

        val baseName = (displayName ?: uri.lastPathSegment ?: "document")
            .substringBeforeLast(".")
            .substringAfterLast("/")
        val sanitized = baseName.replace(Regex("[^A-Za-z0-9_-]"), "_")

        return sanitized.ifBlank { "DOC_${System.currentTimeMillis()}" }
    }

    fun processAndStoreClinicalRecord(pdfUri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            withContext(Dispatchers.Main) {
                isGenerating = true
                _messages.add(
                    Message(
                        text = "Processing clinical record PDF...",
                        isSent = false,
                        messageType = MessageType.TEXT,
                        promptID = promptID++
                    )
                )
            }
            try {
                val rawText = pdfProcessor.extractText(pdfUri)

                if (rawText.isBlank()) {
                    withContext(Dispatchers.Main) {
                        isGenerating = false
                        _messages.add(
                            Message(
                                text = "Error processing clinical record: No readable text found in PDF.",
                                isSent = false,
                                messageType = MessageType.TEXT,
                                promptID = promptID++
                            )
                        )
                    }
                    return@launch
                }

                val textChunks = pdfProcessor.createClinicalChunks(rawText)
                    .map { it.trim() }
                    .filter { it.isNotBlank() }

                if (textChunks.isEmpty()) {
                    withContext(Dispatchers.Main) {
                        isGenerating = false
                        _messages.add(
                            Message(
                                text = "Error processing clinical record: Could not split document into valid chunks.",
                                isSent = false,
                                messageType = MessageType.TEXT,
                                promptID = promptID++
                            )
                        )
                    }
                    return@launch
                }

                if (embedder is PureEmbedder) {
                    Log.w(
                        RAG_TAG,
                        "[INGEST] active embedder='${embedder.backendId}' is the PureEmbedder fallback. " +
                            "Aborting ingestion for this document."
                    )
                    withContext(Dispatchers.Main) {
                        isGenerating = false
                        _messages.add(
                            Message(
                                text = "Cannot index this document: the semantic embedding model isn't loaded " +
                                    "(active backend: ${embedder.backendId}). Please retry once the app finishes initializing.",
                                isSent = false,
                                messageType = MessageType.TEXT,
                                promptID = promptID++
                            )
                        )
                    }
                    return@launch
                }

                val docId = deriveDocumentScopeId(pdfUri)
                Log.i(
                    RAG_TAG,
                    "[INGEST] scoping this document under patientId='$currentPatientId' " +
                        "documentId='$docId' (derived from filename)"
                )

                val currentDate = android.text.format.DateFormat.format("yyyy-MM-dd", java.util.Date()).toString()

                val clinicalChunks = textChunks.map { chunkText ->
                    val contextEnrichedText = """
                    PSYCHIATRIC MEDICAL RECORD EXTRACT
                    $chunkText
                    """.trimIndent()

                    val vector = embedder.generateEmbedding(contextEnrichedText)

                    ClinicalChunk(
                        patientId = currentPatientId,
                        documentId = docId,
                        content = contextEnrichedText,
                        embedding = vector,
                        documentType = "Psychiatric Medical Record",
                        dateString = currentDate
                    )
                }

                clinicalChunks.firstOrNull()?.embedding?.let { sampleVector ->
                    Log.i(
                        RAG_TAG,
                        "[INGEST] embedder='${embedder.backendId}' chunks=${clinicalChunks.size} " +
                            "sample=${sampleVector.embeddingFingerprint()}"
                    )
                }

                vectorRepository.clearChunksForDocument(currentPatientId, docId)
                vectorRepository.insertChunks(clinicalChunks)

                withContext(Dispatchers.Main) {
                    isGenerating = false
                    _messages.add(
                        Message(
                            text = "Medical record updated (${clinicalChunks.size} chunks indexed)",
                            isSent = false,
                            messageType = MessageType.TEXT,
                            promptID = promptID++
                        )
                    )
                }
            } catch (e: Exception) {
                Log.e(RAG_TAG, "[INGEST] failed to process clinical record: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    isGenerating = false
                    _messages.add(
                        Message(
                            text = "Error processing clinical record: ${e.localizedMessage}",
                            isSent = false,
                            messageType = MessageType.TEXT,
                            promptID = promptID++
                        )
                    )
                }
            }
        }
    }

    fun stopGeneration() {
        Log.i("ChatViewModel", "stopGeneration called")
        module?.stop()
    }

    private fun prefillVoxtralAudio(audioFeaturePath: String, textPrompt: String) {
        try {
            val byteData = Files.readAllBytes(Paths.get(audioFeaturePath))
            val buffer = ByteBuffer.wrap(byteData).order(ByteOrder.LITTLE_ENDIAN)
            val floatCount = byteData.size / java.lang.Float.BYTES
            val floats = FloatArray(floatCount)

            for (i in 0 until floatCount) {
                floats[i] = buffer.float
            }
            val bins = 128
            val frames = 3000
            val batchSize = floatCount / (bins * frames)
            val preAudioPrompt = "<s>[INST][BEGIN_AUDIO]"
            val postAudioPrompt = "$textPrompt[/INST]"
            ETLogging.getInstance().log("Voxtral prefill pre-audio prompt: $preAudioPrompt")
            module?.prefillPrompt(preAudioPrompt)
            module?.prefillAudio(floats, batchSize, bins, frames)
            ETLogging.getInstance().log("Voxtral prefill post-audio prompt: $postAudioPrompt")
            module?.prefillPrompt(postAudioPrompt)
        } catch (e: IOException) {
            Log.e("AudioPrefill", "Audio file error")
        }
    }

    fun dismissModelLoadErrorDialog() {
        showModelLoadErrorDialog = false
    }

    fun addSystemMessage(text: String) {
        if (!isDuplicateSystemMessage(text)) {
            _messages.add(Message(text, false, MessageType.SYSTEM, 0))
        }
    }

    private fun isDuplicateSystemMessage(text: String): Boolean {
        if (_messages.isEmpty()) return false
        val lastMessage = _messages.last()
        return lastMessage.messageType == MessageType.SYSTEM && text == lastMessage.text
    }

    override fun onResult(result: String) {
        var processedResult = result

        if (processedResult == PromptFormat.getStopToken(currentSettingsFields.modelType)) {
            module?.stop()
            return
        }

        if (processedResult == "<think>") {
            isInThinkingBlock = true
            return
        }
        if (processedResult == "</think>") {
            isInThinkingBlock = false
            return
        }

        processedResult = PromptFormat.replaceSpecialToken(currentSettingsFields.modelType, processedResult)

        if (currentSettingsFields.modelType == ModelType.LLAMA_3 &&
            processedResult == "<|start_header_id|>"
        ) {
            sawStartHeaderId = true
        }
        if (currentSettingsFields.modelType == ModelType.LLAMA_3 &&
            processedResult == "<|end_header_id|>"
        ) {
            sawStartHeaderId = false
            return
        }
        if (sawStartHeaderId) {
            return
        }

        if (isInThinkingBlock) {
            val keepThinking = !(processedResult == "\n" || processedResult == "\n\n") ||
                    resultMessage?.thinkingContent?.isNotEmpty() == true
            if (keepThinking) {
                resultMessage?.appendThinkingText(processedResult)
            }
        } else {
            val keepResult = !(processedResult == "\n" || processedResult == "\n\n") ||
                    resultMessage?.text?.isNotEmpty() == true
            if (keepResult) {
                resultMessage?.appendText(processedResult)
            }
        }

        val index = _messages.indexOfLast { it === resultMessage }
        if (index >= 0) {
            val updated = resultMessage!!.copy()
            _messages[index] = updated
            resultMessage = updated
        }
    }

    override fun onStats(stats: String) {
        resultMessage?.let { msg ->
            var tps = 0f
            try {
                val jsonObject = JSONObject(stats)
                val numGeneratedTokens = jsonObject.getInt("generated_tokens")
                val inferenceEndMs = jsonObject.getInt("inference_end_ms")
                val promptEvalEndMs = jsonObject.getInt("prompt_eval_end_ms")
                tps = numGeneratedTokens.toFloat() / (inferenceEndMs - promptEvalEndMs) * 1000
            } catch (e: JSONException) {
                Log.e("LLM", "Error parsing JSON: ${e.message}")
            }
            msg.tokensPerSecond = tps
            val index = _messages.indexOfLast { it === msg }
            if (index >= 0) {
                val updated = msg.copy()
                _messages[index] = updated
                resultMessage = updated
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        (embedder as? java.io.Closeable)?.close()
    }

    companion object {
        private const val MAX_NUM_OF_IMAGES = 5
        private const val RAG_TAG = "ChatViewModel-RAG"
    }
}
