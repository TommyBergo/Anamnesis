/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo

import android.content.Context
import android.util.Log
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.hasContentDescription
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextClearance
import androidx.compose.ui.test.performTextInput
import androidx.test.core.app.ApplicationProvider
import androidx.test.espresso.Espresso
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Ignore
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** UI workflow test simulating the model loading and chat workflow (model/tokenizer selection, sending messages, stopping generation) via Compose testing APIs. */
@RunWith(AndroidJUnit4::class)
@LargeTest
class UIWorkflowTest {

    companion object {
        private const val TAG = "UIWorkflowTest"
        private const val RESPONSE_TAG = "LLAMA_RESPONSE"
        private const val DEFAULT_MODEL_FILE = "stories110M.pte"
        private const val DEFAULT_TOKENIZER_FILE = "tokenizer.model"
    }

    @get:Rule
    val composeTestRule = createAndroidComposeRule<WelcomeActivity>()

    private lateinit var modelFile: String
    private lateinit var tokenizerFile: String

    @Before
    fun setUp() {
        val args = InstrumentationRegistry.getArguments()
        modelFile = args.getString("modelFile", DEFAULT_MODEL_FILE) ?: DEFAULT_MODEL_FILE
        tokenizerFile = args.getString("tokenizerFile", DEFAULT_TOKENIZER_FILE) ?: DEFAULT_TOKENIZER_FILE
        Log.i(TAG, "Using model: $modelFile, tokenizer: $tokenizerFile")

        val context = ApplicationProvider.getApplicationContext<Context>()
        val prefs = context.getSharedPreferences(
            context.getString(R.string.demo_pref_file_key),
            Context.MODE_PRIVATE
        )
        prefs.edit().clear().commit()
    }

    private fun clearChatHistory() {
        composeTestRule.waitForIdle()

        try {
            composeTestRule.onNodeWithText("App Settings").performClick()
            composeTestRule.waitUntil(timeoutMillis = 3000) {
                composeTestRule.onAllNodesWithText("Clear Conversation History")
                    .fetchSemanticsNodes().isNotEmpty()
            }
        } catch (e: Exception) {
            Log.d(TAG, "Could not open App Settings to clear history: ${e.message}")
            return
        }

        try {
            composeTestRule.onNodeWithText("Clear Conversation History").performClick()
            composeTestRule.waitUntil(timeoutMillis = 3000) {
                composeTestRule.onAllNodesWithText("Clear").fetchSemanticsNodes().isNotEmpty()
            }
            composeTestRule.onNodeWithText("Clear").performClick()
            composeTestRule.waitForIdle()
            Log.i(TAG, "Chat history cleared")
        } catch (e: Exception) {
            Log.d(TAG, "Could not clear chat history: ${e.message}")
        }

        try {
            composeTestRule.onNodeWithContentDescription("Back").performClick()
            composeTestRule.waitForIdle()
        } catch (e: Exception) {
            Log.d(TAG, "Could not press back after clearing history: ${e.message}")
        }
    }

    private fun loadModel(): Boolean {
        composeTestRule.onNodeWithText("Load local model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5001) {
            composeTestRule.onAllNodesWithText("Select a Model").fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText("Model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5002) {
            composeTestRule.onAllNodesWithText("Select model path").fetchSemanticsNodes().isNotEmpty()
        }

        try {
            composeTestRule.onNodeWithText(modelFile, substring = true).performClick()
        } catch (e: AssertionError) {
            Log.e(TAG, "Model file not found: $modelFile")
            return false
        }

        composeTestRule.onNodeWithText("Tokenizer").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5003) {
            composeTestRule.onAllNodesWithText("Select tokenizer path").fetchSemanticsNodes().isNotEmpty()
        }

        try {
            composeTestRule.onNodeWithText(tokenizerFile, substring = true).performClick()
        } catch (e: AssertionError) {
            Log.e(TAG, "Tokenizer file not found: $tokenizerFile")
            return false
        }

        composeTestRule.onNodeWithText("Load Model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5004) {
            composeTestRule.onAllNodesWithText("Yes").fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText("Yes").performClick()

        return true
    }

    private fun waitForModelLoaded(timeoutMs: Long = 60000): Boolean {
        return try {
            var wasSuccess = false
            composeTestRule.waitUntil(timeoutMillis = timeoutMs) {
                val successNodes = composeTestRule.onAllNodesWithText("Successfully loaded", substring = true)
                    .fetchSemanticsNodes()
                val errorNodes = composeTestRule.onAllNodesWithText("Model load failure", substring = true)
                    .fetchSemanticsNodes()
                wasSuccess = successNodes.isNotEmpty()
                successNodes.isNotEmpty() || errorNodes.isNotEmpty()
            }
            if (wasSuccess) {
                Log.i(TAG, "Model loaded successfully")
            } else {
                Log.e(TAG, "Model load failed")
            }
            wasSuccess
        } catch (e: Exception) {
            Log.e(TAG, "Model loading timed out after ${timeoutMs}ms: ${e.message}")
            false
        }
    }

    private fun typeInChatInput(text: String) {
        composeTestRule.onNodeWithTag("chat_input_field").performClick()
        composeTestRule.waitForIdle()
        composeTestRule.onNodeWithTag("chat_input_field").performTextInput(text)
        composeTestRule.waitForIdle()
    }

    private fun clearChatInput() {
        composeTestRule.onNodeWithTag("chat_input_field").performTextClearance()
        composeTestRule.waitForIdle()
    }

    private fun assertModelResponseNotEmpty(timeoutMs: Long = 10000) {
        try {
            composeTestRule.waitUntil(timeoutMillis = timeoutMs) {
                val tpsNodes = composeTestRule.onAllNodesWithText("t/s", substring = true)
                    .fetchSemanticsNodes()
                val tokpsNodes = composeTestRule.onAllNodesWithText("tok/s", substring = true)
                    .fetchSemanticsNodes()
                tpsNodes.isNotEmpty() || tokpsNodes.isNotEmpty()
            }
            Log.i(TAG, "Model response verified - found generation metrics")
        } catch (e: Exception) {
            throw AssertionError("Model response appears to be empty - no generation metrics found after ${timeoutMs}ms")
        }
    }

    private fun logModelResponse() {
        try {
            Log.i(RESPONSE_TAG, "BEGIN_RESPONSE")
            val responseNodes = composeTestRule.onAllNodesWithText("t/s", substring = true)
                .fetchSemanticsNodes()
            for (node in responseNodes) {
                val text = node.config.getOrElse(SemanticsProperties.Text) { emptyList() }
                    .joinToString(" ") { it.text }
                if (text.isNotBlank()) {
                    Log.i(RESPONSE_TAG, text)
                }
            }
            Log.i(RESPONSE_TAG, "END_RESPONSE")
        } catch (e: Exception) {
            Log.d(TAG, "Could not log model response: ${e.message}")
        }
    }

    @Test
    fun testModelLoadingWorkflow() {
        composeTestRule.waitForIdle()

        composeTestRule.onNodeWithText("Load local model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5005) {
            composeTestRule.onAllNodesWithText("Select a Model").fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText("Select a Model").assertIsDisplayed()
        composeTestRule.onNodeWithText("Load Model").assertIsDisplayed()
        composeTestRule.onNodeWithText("no model selected").assertIsDisplayed()
        composeTestRule.onNodeWithText("no tokenizer selected").assertIsDisplayed()

        composeTestRule.onNodeWithText("Model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5006) {
            composeTestRule.onAllNodesWithText("Select model path").fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText(modelFile, substring = true).performClick()

        composeTestRule.onNodeWithText("Tokenizer").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5007) {
            composeTestRule.onAllNodesWithText("Select tokenizer path").fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText(tokenizerFile, substring = true).performClick()

        composeTestRule.onNodeWithText("Load Model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5008) {
            composeTestRule.onAllNodesWithText("Yes").fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText("Yes").performClick()
    }

    @Test
    fun testSendMessageAndReceiveResponse() {
        composeTestRule.waitForIdle()

        clearChatHistory()

        val loaded = loadModel()
        assertTrue("Model should be selected successfully", loaded)

        val modelLoaded = waitForModelLoaded(90000)
        assertTrue("Model should be loaded successfully", modelLoaded)

        typeInChatInput("tell me a story")

        composeTestRule.waitUntil(timeoutMillis = 5025) {
            try {
                composeTestRule.onNodeWithContentDescription("Send").assertIsEnabled()
                true
            } catch (e: AssertionError) {
                Log.d(TAG, "Send button not yet enabled: ${e.message}")
                false
            }
        }

        composeTestRule.onNodeWithContentDescription("Send").performClick()
        composeTestRule.waitForIdle()

        val generationComplete = waitForGenerationComplete()
        assertTrue("Generation should complete", generationComplete)

        assertModelResponseNotEmpty()

        logModelResponse()

        Log.i(TAG, "Send message and receive response test completed successfully")
    }

    @Test
    fun testStopGeneration() {
        composeTestRule.waitForIdle()

        clearChatHistory()

        val loaded = loadModel()
        assertTrue("Model should be selected successfully", loaded)

        val modelLoaded = waitForModelLoaded(90000)
        assertTrue("Model should be loaded successfully", modelLoaded)

        typeInChatInput("Write a very long story about a brave knight who goes on an adventure")

        composeTestRule.waitUntil(timeoutMillis = 5026) {
            try {
                composeTestRule.onNodeWithContentDescription("Send").assertIsEnabled()
                true
            } catch (e: AssertionError) {
                Log.d(TAG, "Send button not yet enabled: ${e.message}")
                false
            }
        }

        composeTestRule.onNodeWithContentDescription("Send").performClick()
        composeTestRule.waitForIdle()

        try {
            composeTestRule.waitUntil(timeoutMillis = 5009) {
                composeTestRule.onAllNodes(hasContentDescription("Stop"))
                    .fetchSemanticsNodes().isNotEmpty()
            }
        } catch (e: Exception) {
            Log.i(TAG, "Stop button not found - generation may have completed")
        }

        Thread.sleep(500)

        composeTestRule.onNodeWithContentDescription("Stop").performClick()

        composeTestRule.waitForIdle()

        waitForGenerationComplete(30000)

        assertModelResponseNotEmpty()

        Log.i(TAG, "Stop generation test completed successfully")
    }

    @Test
    fun testEmptyPromptSend() {
        composeTestRule.waitForIdle()

        val loaded = loadModel()
        assertTrue("Model should be selected successfully", loaded)

        val modelLoaded = waitForModelLoaded(90000)
        assertTrue("Model should be loaded successfully", modelLoaded)

        composeTestRule.waitUntil(timeoutMillis = 5010) {
            composeTestRule.onAllNodes(hasContentDescription("Send"))
                .fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithContentDescription("Send").assertIsNotEnabled()

        typeInChatInput("hello")

        composeTestRule.waitUntil(timeoutMillis = 2001) {
            try {
                composeTestRule.onNodeWithContentDescription("Send").assertIsEnabled()
                true
            } catch (e: AssertionError) {
                false
            }
        }

        composeTestRule.onNodeWithContentDescription("Send").assertIsEnabled()

        clearChatInput()

        composeTestRule.waitUntil(timeoutMillis = 2002) {
            try {
                composeTestRule.onNodeWithContentDescription("Send").assertIsNotEnabled()
                true
            } catch (e: AssertionError) {
                false
            }
        }

        composeTestRule.onNodeWithContentDescription("Send").assertIsNotEnabled()
    }

    @Test
    fun testNoFilesInDirectory() {
        composeTestRule.waitForIdle()

        composeTestRule.onNodeWithText("Load local model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5011) {
            composeTestRule.onAllNodesWithText("Select a Model").fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText("Select a Model").assertIsDisplayed()

        composeTestRule.onNodeWithText("Model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5012) {
            composeTestRule.onAllNodesWithText("Select model path").fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText("Select model path").assertIsDisplayed()

        try {
            composeTestRule.onNodeWithText("Cancel").performClick()
        } catch (e: AssertionError) {
            composeTestRule.onNodeWithText(modelFile, substring = true).performClick()
        }
        composeTestRule.waitForIdle()
    }

    @Test
    fun testCancelFileSelection() {
        composeTestRule.waitForIdle()

        composeTestRule.onNodeWithText("Load local model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5013) {
            composeTestRule.onAllNodesWithText("Select a Model").fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText("no model selected").assertIsDisplayed()
        composeTestRule.onNodeWithText("no tokenizer selected").assertIsDisplayed()

        composeTestRule.onNodeWithText("Model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5014) {
            composeTestRule.onAllNodesWithText("Select model path").fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithText(modelFile, substring = true).performClick()

        composeTestRule.waitUntil(timeoutMillis = 5015) {
            composeTestRule.onAllNodesWithText(modelFile, substring = true)
                .fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText(modelFile, substring = true).assertIsDisplayed()

        composeTestRule.onNodeWithText("Model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5016) {
            composeTestRule.onAllNodesWithText("Select model path").fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText("Cancel").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5017) {
            composeTestRule.onAllNodesWithText("Select model path").fetchSemanticsNodes().isEmpty()
        }

        composeTestRule.onNodeWithText(modelFile, substring = true).assertIsDisplayed()
    }

    @Test
    fun testLoadButtonDisabledState() {
        composeTestRule.waitForIdle()

        composeTestRule.onNodeWithText("Load local model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5018) {
            composeTestRule.onAllNodesWithText("Select a Model").fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText("Load Model").assertIsNotEnabled()

        composeTestRule.onNodeWithText("Model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5019) {
            composeTestRule.onAllNodesWithText("Select model path").fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithText(modelFile, substring = true).performClick()

        composeTestRule.waitUntil(timeoutMillis = 5020) {
            composeTestRule.onAllNodesWithText("Select model path").fetchSemanticsNodes().isEmpty()
        }

        composeTestRule.onNodeWithText("Load Model").assertIsNotEnabled()

        composeTestRule.onNodeWithText("Tokenizer").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5021) {
            composeTestRule.onAllNodesWithText("Select tokenizer path").fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithText(tokenizerFile, substring = true).performClick()

        composeTestRule.waitUntil(timeoutMillis = 5022) {
            composeTestRule.onAllNodesWithText("Select tokenizer path").fetchSemanticsNodes().isEmpty()
        }

        composeTestRule.onNodeWithText("Load Model").assertIsEnabled()
    }

    @Test
    fun testWhitespaceOnlyPrompt() {
        composeTestRule.waitForIdle()

        val loaded = loadModel()
        assertTrue("Model should be selected successfully", loaded)

        val modelLoaded = waitForModelLoaded(90000)
        assertTrue("Model should be loaded successfully", modelLoaded)

        composeTestRule.waitUntil(timeoutMillis = 5023) {
            composeTestRule.onAllNodes(hasContentDescription("Send"))
                .fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithContentDescription("Send").assertIsNotEnabled()

        typeInChatInput("     ")

        composeTestRule.onNodeWithContentDescription("Send").assertIsNotEnabled()

        clearChatInput()
        typeInChatInput("hello")

        composeTestRule.waitUntil(timeoutMillis = 2003) {
            try {
                composeTestRule.onNodeWithContentDescription("Send").assertIsEnabled()
                true
            } catch (e: AssertionError) {
                false
            }
        }

        composeTestRule.onNodeWithContentDescription("Send").assertIsEnabled()
    }

    @Ignore("Temporarily disabled")
    @Test
    fun testMultipleMessagesConversation() {
        composeTestRule.waitForIdle()

        clearChatHistory()

        val loaded = loadModel()
        assertTrue("Model should be selected successfully", loaded)

        val modelLoaded = waitForModelLoaded(90000)
        assertTrue("Model should be loaded successfully", modelLoaded)

        val firstMessage = "XYZTEST123"
        typeInChatInput(firstMessage)
        composeTestRule.onNodeWithContentDescription("Send").performClick()
        composeTestRule.waitForIdle()

        val firstResponseComplete = waitForGenerationComplete(120000)
        assertTrue("First response should complete", firstResponseComplete)

        composeTestRule.onNodeWithText(firstMessage, substring = true).assertExists()
        assertModelResponseNotEmpty()

        val secondMessage = "ABCTEST456"
        typeInChatInput(secondMessage)

        composeTestRule.waitUntil(timeoutMillis = 5024) {
            try {
                composeTestRule.onNodeWithContentDescription("Send").assertIsEnabled()
                true
            } catch (e: AssertionError) {
                false
            }
        }

        composeTestRule.onNodeWithContentDescription("Send").performClick()
        composeTestRule.waitForIdle()

        val secondResponseComplete = waitForGenerationComplete(120001)
        assertTrue("Second response should complete", secondResponseComplete)

        composeTestRule.onNodeWithText(firstMessage, substring = true).assertExists()
        composeTestRule.onNodeWithText(secondMessage, substring = true).assertExists()

        assertModelResponseNotEmpty()

        Log.i(TAG, "Multiple messages conversation test completed successfully")
    }

    private fun waitForGenerationComplete(timeoutMs: Long = 120000): Boolean {
        return try {
            composeTestRule.waitUntil(timeoutMillis = timeoutMs) {
                val tpsNodes = composeTestRule.onAllNodesWithText("t/s", substring = true)
                    .fetchSemanticsNodes()
                val tokpsNodes = composeTestRule.onAllNodesWithText("tok/s", substring = true)
                    .fetchSemanticsNodes()
                tpsNodes.isNotEmpty() || tokpsNodes.isNotEmpty()
            }
            Log.i(TAG, "Generation complete - found generation metrics")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Generation timed out after ${timeoutMs}ms")
            false
        }
    }

    @Test
    fun testCollapseMediaButton() {
        composeTestRule.waitForIdle()

        try {
            composeTestRule.onNodeWithContentDescription("Add media").assertIsDisplayed()

            composeTestRule.onNodeWithContentDescription("Add media").performClick()

            composeTestRule.waitUntil(timeoutMillis = 3000) {
                composeTestRule.onAllNodesWithText("Gallery").fetchSemanticsNodes().isNotEmpty()
            }

            composeTestRule.onNodeWithText("Gallery").assertIsDisplayed()
            composeTestRule.onNodeWithText("Camera").assertIsDisplayed()

            composeTestRule.onNodeWithContentDescription("Collapse media").performClick()

            composeTestRule.waitUntil(timeoutMillis = 3000) {
                composeTestRule.onAllNodesWithText("Gallery").fetchSemanticsNodes().isEmpty()
            }

            composeTestRule.onNodeWithText("Gallery").assertDoesNotExist()
        } catch (e: AssertionError) {
            Log.i(TAG, "Media buttons not present - might be MediaTek backend")
        }
    }

    @Test
    fun testWelcomeScreenNavigation() {
        composeTestRule.waitForIdle()

        composeTestRule.onNodeWithText("ExecuTorch Llama Demo").assertIsDisplayed()
        composeTestRule.onNodeWithText("Welcome to ExecuTorch Llama Demo").assertIsDisplayed()
        composeTestRule.onNodeWithText("Load local model").assertIsDisplayed()
        composeTestRule.onNodeWithText("App Settings").assertIsDisplayed()

        composeTestRule.onNodeWithText("App Settings").performClick()
        composeTestRule.waitUntil(timeoutMillis = 3000) {
            composeTestRule.onAllNodesWithText("App Settings", useUnmergedTree = true)
                .fetchSemanticsNodes().size >= 1
        }

        composeTestRule.onNodeWithText("Appearance").assertIsDisplayed()
        composeTestRule.onNodeWithText("Theme").assertIsDisplayed()
        composeTestRule.onNodeWithText("Clear Conversation History").assertIsDisplayed()

        composeTestRule.onNodeWithContentDescription("Back").performClick()
        composeTestRule.waitUntil(timeoutMillis = 3000) {
            composeTestRule.onAllNodesWithText("ExecuTorch Llama Demo").fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText("ExecuTorch Llama Demo").assertIsDisplayed()
        composeTestRule.onNodeWithText("Load local model").assertIsDisplayed()

        composeTestRule.onNodeWithText("Load local model").performClick()
        composeTestRule.waitUntil(timeoutMillis = 3000) {
            composeTestRule.onAllNodesWithText("Select a Model").fetchSemanticsNodes().isNotEmpty()
        }

        composeTestRule.onNodeWithText("Select a Model").assertIsDisplayed()
        composeTestRule.onNodeWithText("Backend").assertIsDisplayed()
        composeTestRule.onNodeWithText("Load Model").assertIsDisplayed()

        Log.i(TAG, "Welcome screen navigation test completed successfully")
    }
}
