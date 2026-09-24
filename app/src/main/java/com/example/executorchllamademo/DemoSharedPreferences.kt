/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo

import android.content.Context
import android.content.SharedPreferences
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

/** Persists and retrieves app settings, module settings, chat messages, and logs via SharedPreferences. */
class DemoSharedPreferences(private val context: Context) {

    private val sharedPreferences: SharedPreferences = context.getSharedPreferences(
        context.getString(R.string.demo_pref_file_key),
        Context.MODE_PRIVATE
    )

    private val gson = Gson()

    fun getSavedMessages(): String {
        return sharedPreferences.getString(
            context.getString(R.string.saved_messages_json_key),
            ""
        ) ?: ""
    }

    fun addMessages(messages: List<Message>) {
        val editor = sharedPreferences.edit()
        val msgJSON = gson.toJson(messages)
        editor.putString(context.getString(R.string.saved_messages_json_key), msgJSON)
        editor.apply()
    }

    fun removeExistingMessages() {
        val editor = sharedPreferences.edit()
        editor.remove(context.getString(R.string.saved_messages_json_key))
        editor.apply()
    }

    fun getAppSettings(): AppSettings {
        val json = sharedPreferences.getString(PREF_KEY_APP_SETTINGS, null)
        return if (json.isNullOrEmpty()) {
            AppSettings()
        } else {
            try {
                val parsed = gson.fromJson(json, AppSettings::class.java) ?: AppSettings()
                if (parsed.embedderBackend == null) parsed.copy(embedderBackend = EmbedderBackend.EMBEDDING_GEMMA) else parsed
            } catch (e: Exception) {
                AppSettings()
            }
        }
    }

    fun saveAppSettings(settings: AppSettings) {
        val editor = sharedPreferences.edit()
        editor.putString(PREF_KEY_APP_SETTINGS, gson.toJson(settings))
        editor.apply()
    }

    fun getModuleSettings(): ModuleSettings {
        val json = sharedPreferences.getString(PREF_KEY_MODULE_SETTINGS, null)
        return if (json.isNullOrEmpty()) {
            ModuleSettings()
        } else {
            try {
                gson.fromJson(json, ModuleSettings::class.java) ?: ModuleSettings()
            } catch (e: Exception) {
                ModuleSettings()
            }
        }
    }

    fun saveModuleSettings(settings: ModuleSettings) {
        val editor = sharedPreferences.edit()
        editor.putString(PREF_KEY_MODULE_SETTINGS, gson.toJson(settings))
        editor.apply()
    }

    fun saveLogs() {
        val editor = sharedPreferences.edit()
        val logsCopy = ArrayList(ETLogging.getInstance().getLogs())
        val msgJSON = gson.toJson(logsCopy)
        editor.putString(context.getString(R.string.logs_json_key), msgJSON)
        editor.apply()
    }

    fun removeExistingLogs() {
        val editor = sharedPreferences.edit()
        editor.remove(context.getString(R.string.logs_json_key))
        editor.apply()
    }

    fun getSavedLogs(): ArrayList<AppLog> {
        val logsJSONString = sharedPreferences.getString(
            context.getString(R.string.logs_json_key),
            null
        )
        if (logsJSONString.isNullOrEmpty()) {
            return ArrayList()
        }
        val type = object : TypeToken<ArrayList<AppLog>>() {}.type
        return gson.fromJson(logsJSONString, type) ?: ArrayList()
    }

    companion object {
        private const val PREF_KEY_APP_SETTINGS = "app_settings_json"
        private const val PREF_KEY_MODULE_SETTINGS = "module_settings_json"
    }
}
