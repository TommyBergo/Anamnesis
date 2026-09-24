/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo.ui.viewmodel

import androidx.compose.runtime.mutableStateListOf
import androidx.lifecycle.ViewModel
import com.example.executorchllamademo.AppLog
import com.example.executorchllamademo.ETLogging

/** View model that loads, exposes, and clears the application's in-memory log buffer for the logs screen. */
class LogsViewModel : ViewModel() {
    private val _logs = mutableStateListOf<AppLog>()
    val logs: List<AppLog> = _logs

    fun loadLogs() {
        _logs.clear()
        _logs.addAll(ETLogging.getInstance().getLogs())
    }

    fun clearLogs() {
        ETLogging.getInstance().clearLogs()
        _logs.clear()
    }

    fun saveLogs() {
        ETLogging.getInstance().saveLogs()
    }
}
