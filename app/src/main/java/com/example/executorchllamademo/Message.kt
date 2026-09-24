/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

/** Represents a mutable chat message in the conversation, along with its generation stats and RAG debug info. */
class Message(
    text: String,
    isSent: Boolean,
    val messageType: MessageType,
    val promptID: Int,
    existingTimestamp: Long? = null,
    val id: String = UUID.randomUUID().toString()
) {
    @get:JvmName("getIsSent")
    val isSent: Boolean = isSent
    var text: String = if (messageType == MessageType.IMAGE) "" else text

    var imagePath: String? = if (messageType == MessageType.IMAGE) text else null
        private set

    val timestamp: Long = existingTimestamp
        ?: if (messageType != MessageType.SYSTEM) System.currentTimeMillis() else 0L

    var tokensPerSecond: Float = 0f

    var totalGenerationTime: Long = 0L

    var thinkingContent: String = ""

    var ragDebugHeader: String = ""

    fun appendText(text: String) {
        this.text += text
    }

    fun appendThinkingText(text: String) {
        thinkingContent += text
    }

    fun copy(): Message {
        val sourceText = if (messageType == MessageType.IMAGE) (imagePath ?: "") else text
        return Message(sourceText, isSent, messageType, promptID, timestamp, id).also {
            it.tokensPerSecond = tokensPerSecond
            it.totalGenerationTime = totalGenerationTime
            it.thinkingContent = thinkingContent
            it.ragDebugHeader = ragDebugHeader
        }
    }

    fun getFormattedTimestamp(): String {
        val formatter = SimpleDateFormat(TIMESTAMP_FORMAT, Locale.getDefault())
        val date = Date(timestamp)
        return formatter.format(date)
    }

    companion object {
        private const val TIMESTAMP_FORMAT = "hh:mm a"
    }
}
