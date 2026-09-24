/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("kotlin-kapt")
}
apply(plugin = "io.objectbox")
// Model files configuration for instrumentation tests
// Supported presets: stories, llama, qwen3, custom
val modelPreset: String = (project.findProperty("modelPreset") as? String) ?: "stories"

// Preset configurations
val modelPresets = mapOf(
  "stories" to mapOf(
    "baseUrl" to "https://ossci-android.s3.amazonaws.com/executorch/stories/snapshot-20260114",
    "pteFile" to "stories110M.pte",
    "tokenizerFile" to "tokenizer.model",
    "verifyChecksum" to true
  ),
  "llama" to mapOf(
    "baseUrl" to "https://huggingface.co/executorch-community/Llama-3.2-1B-ET/resolve/main",
    "pteFile" to "llama3_2-1B.pte",
    "tokenizerFile" to "tokenizer.model",
    "verifyChecksum" to false
  ),
  "qwen3" to mapOf(
    "baseUrl" to "https://huggingface.co/pytorch/Qwen3-4B-INT8-INT4/resolve/main",
    "pteFile" to "model.pte",
    "tokenizerFile" to "tokenizer.json",
    "verifyChecksum" to false
  )
)

// Custom URLs (used when modelPreset is "custom")
val customPteUrl: String? = project.findProperty("customPteUrl") as? String
val customTokenizerUrl: String? = project.findProperty("customTokenizerUrl") as? String

val deviceModelDir = "/data/local/tmp/llama"
val skipModelDownload: Boolean = (project.findProperty("skipModelDownload") as? String)?.toBoolean() ?: false

fun execCmd(vararg args: String): String {
  val process = ProcessBuilder(*args)
    .redirectErrorStream(true)
    .start()
  val output = process.inputStream.bufferedReader().readText().trim()
  process.waitFor()
  return output
}

fun execCmdWithExitCode(vararg args: String): Pair<Int, String> {
  val process = ProcessBuilder(*args)
    .redirectErrorStream(true)
    .start()
  val output = process.inputStream.bufferedReader().readText().trim()
  val exitCode = process.waitFor()
  return Pair(exitCode, output)
}

// Streaming version that shows output in real-time (for long-running commands)
fun execCmdStreaming(vararg args: String): Int {
  val process = ProcessBuilder(*args)
    .inheritIO()
    .start()
  return process.waitFor()
}

tasks.register("pushModelFiles") {
  description = "Download model files and push to connected Android device if not present"
  group = "verification"

  doLast {
    if (skipModelDownload) {
      logger.lifecycle("Skipping model download (skipModelDownload=true)")
      return@doLast
    }

    logger.lifecycle("Using model preset: $modelPreset")

    // Determine URLs based on preset
    val pteUrl: String
    val tokenizerUrl: String
    val verifyChecksum: Boolean

    if (modelPreset == "custom") {
      pteUrl = customPteUrl ?: throw GradleException("customPteUrl is required when modelPreset is 'custom'")
      tokenizerUrl = customTokenizerUrl ?: throw GradleException("customTokenizerUrl is required when modelPreset is 'custom'")
      verifyChecksum = false
    } else {
      val preset = modelPresets[modelPreset] ?: throw GradleException("Unknown model preset: $modelPreset. Valid options: ${modelPresets.keys.joinToString(", ")}, custom")
      val baseUrl = preset["baseUrl"] as String
      pteUrl = "$baseUrl/${preset["pteFile"]}"
      tokenizerUrl = "$baseUrl/${preset["tokenizerFile"]}"
      verifyChecksum = preset["verifyChecksum"] as Boolean
    }

    // Files to download: source URL -> target name on device (keep original filenames)
    val filesToDownload = mapOf(
      pteUrl to pteUrl.substringAfterLast("/"),
      tokenizerUrl to tokenizerUrl.substringAfterLast("/")
    )

    // Check if adb is available
    val adbPath = android.adbExecutable.absolutePath
    val (adbCheckCode, _) = execCmdWithExitCode(adbPath, "devices")
    if (adbCheckCode != 0) {
      throw GradleException("adb is not available or no device connected")
    }

    // Check which files need to be pushed
    val filesToPush = filesToDownload.filter { (_, targetName) ->
      val devicePath = "$deviceModelDir/$targetName"
      val (exitCode, _) = execCmdWithExitCode(adbPath, "shell", "test -f $devicePath && echo exists")
      exitCode != 0
    }

    if (filesToPush.isEmpty()) {
      logger.lifecycle("All model files already present on device")
      return@doLast
    }

    logger.lifecycle("Need to push ${filesToPush.size} model file(s): ${filesToPush.values.joinToString(", ")}")

    // Create temp directory using mktemp
    val tempDir = execCmd("mktemp", "-d")
    logger.lifecycle("Using temp directory: $tempDir")

    try {
      // Create device directory
      execCmd(adbPath, "shell", "mkdir -p $deviceModelDir")

      for ((sourceUrl, targetName) in filesToPush) {
        val localPath = "$tempDir/$targetName"
        val devicePath = "$deviceModelDir/$targetName"

        // Download file with progress indicator
        logger.lifecycle("Downloading from $sourceUrl...")
        val dlCode = execCmdStreaming("curl", "-fL", "--progress-bar", "-o", localPath, sourceUrl)
        if (dlCode != 0) {
          throw GradleException("Failed to download from $sourceUrl")
        }

        // Verify checksum if enabled and available (only for stories preset)
        if (verifyChecksum && modelPreset == "stories") {
          val sourceName = sourceUrl.substringAfterLast("/")
          val checksumPath = "$tempDir/$sourceName.sha256sums"
          val checksumUrl = "$sourceUrl.sha256sums"

          logger.lifecycle("Verifying checksum for $sourceName...")
          val (csDownloadCode, _) = execCmdWithExitCode(
            "curl", "-fL", "-o", checksumPath, checksumUrl
          )
          if (csDownloadCode == 0) {
            // Copy file to original name for checksum verification if needed
            val tempForChecksum = "$tempDir/$sourceName"
            val needsCopy = localPath != tempForChecksum
            if (needsCopy) {
              execCmd("cp", localPath, tempForChecksum)
            }

            val (verifyCode, verifyOutput) = execCmdWithExitCode(
              "bash", "-c", "cd $tempDir && sha256sum -c $sourceName.sha256sums"
            )
            if (verifyCode != 0) {
              throw GradleException("Checksum verification failed for $sourceName: $verifyOutput")
            }
            logger.lifecycle("Checksum verified for $sourceName")
            // Only delete the temp copy if we made one
            if (needsCopy) {
              execCmd("rm", "-f", tempForChecksum)
            }
          } else {
            logger.lifecycle("Checksum file not available, skipping verification")
          }
        }

        // Push to device with progress
        logger.lifecycle("Pushing $targetName to device...")
        val pushCode = execCmdStreaming(adbPath, "push", localPath, devicePath)
        if (pushCode != 0) {
          throw GradleException("Failed to push $targetName to device")
        }
        logger.lifecycle("Successfully pushed $targetName")
      }
    } finally {
      // Clean up temp directory
      logger.lifecycle("Cleaning up temp directory...")
      execCmd("rm", "-rf", tempDir)
    }

    logger.lifecycle("All model files pushed successfully")
  }
}

// Make all connected Android test tasks depend on pushModelFiles
tasks.whenTaskAdded {
  if (name.startsWith("connected") && name.endsWith("AndroidTest")) {
    dependsOn("pushModelFiles")
  }
}

val useLocalAar: Boolean? = (project.findProperty("useLocalAar") as? String)?.toBoolean()

android {
    namespace = "com.example.executorchllamademo"
    compileSdk = 35
    // Pinned for build reproducibility - AGP's default NDK resolution can otherwise pick a
    // different version per machine, or try to auto-download one when building offline.
    ndkVersion = "25.0.8775105"

    defaultConfig {
        applicationId = "com.example.executorchllamademo"
        testApplicationId = "com.example.executorchllamademo.test"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Automatically set instrumentation arguments based on model preset
        val preset = modelPresets[modelPreset]
        if (preset != null) {
            testInstrumentationRunnerArguments["modelFile"] = preset["pteFile"] as String
            testInstrumentationRunnerArguments["tokenizerFile"] = preset["tokenizerFile"] as String
        }

        vectorDrawables { useSupportLibrary = true }
        // USearch's prebuilt libusearch_c.so (app/src/main/jniLibs/) is arm64-v8a only - restrict
        // the native build to that ABI so other architectures don't fail looking for it.
        externalNativeBuild { cmake { cppFlags += ""; abiFilters += "arm64-v8a" } }
    }

    externalNativeBuild {
        cmake { path = file("src/main/cpp/CMakeLists.txt") }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
    }
    androidResources {
        noCompress.add("tflite")
        // MediaPipe Tasks needs its .task bundles (themselves ZIP archives of a .tflite +
        // sentencepiece model + metadata) stored uncompressed so they can be opened/mmap'd
        // directly from the APK at runtime, same reasoning as the .tflite exclusion above.
        noCompress.add("task")
    }
    packaging { 
        resources { 
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
            excludes += "/META-INDEX"
            excludes += "/META-NOTICES"
            excludes += "META-INF/DEPENDENCIES"
            excludes += "META-INF/LICENSE"
            excludes += "META-INF/LICENSE.txt"
            excludes += "META-INF/NOTICE"
            excludes += "META-INF/NOTICE.txt"
        }
        resources.excludes.add("META-INF/native/lib/linux/x86_64/libtokenizers.so")
        resources.excludes.add("META-INF/native/lib/osx/x86_64/libtokenizers.so")
        resources.excludes.add("META-INF/native/lib/windows/x86_64/libtokenizers.dll")
        jniLibs {
            keepDebugSymbols += "**/*.so"
            // AGP defaults useLegacyPackaging to false above minSdk 23 (this app's minSdk is 28),
            // which stores .so files uncompressed in the APK and loads them via mmap without ever
            // extracting to ApplicationInfo.nativeLibraryDir on disk. SqliteVecVectorRepository
            // needs a real on-disk path to hand sqlite-vec's libvec0.so to SQLiteCustomExtension's
            // dlopen() (not System.loadLibrary(), which would tolerate the in-APK case) - force
            // legacy packaging so the library actually gets extracted at install time.
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material.icons.extended)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation("io.coil-kt:coil-compose:2.4.0")
    implementation("androidx.camera:camera-core:1.3.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.0")
    implementation("com.facebook.fbjni:fbjni:0.5.1")
    implementation("com.google.code.gson:gson:2.8.6")
    implementation("com.halilibo.compose-richtext:richtext-commonmark:1.0.0-alpha02")
    implementation("com.halilibo.compose-richtext:richtext-ui-material3:1.0.0-alpha02")
    if (useLocalAar == true) {
        implementation(files("libs/executorch.aar"))
    } else {
        implementation("org.pytorch:executorch-android:1.1.0")
        // https://mvnrepository.com/artifact/org.pytorch/executorch-android-qnn
        // Uncomment this to enable QNN
        // implementation("org.pytorch:executorch-android-qnn:1.1.0")

        // https://mvnrepository.com/artifact/org.pytorch/executorch-android-vulkan
        // uncomment to enable vulkan
        // implementation("org.pytorch:executorch-android-vulkan:1.1.0")
    }
    implementation("androidx.activity:activity:1.9.0")
    implementation("org.json:json:20250107")
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation("androidx.test.uiautomator:uiautomator:2.2.0")
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
    implementation("io.objectbox:objectbox-kotlin:4.0.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")
    implementation("com.google.mlkit:text-recognition:16.0.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.8.1")
    // LiteRT (Google's rebrand/continuation of TensorFlow Lite, same org.tensorflow.lite.*
    // Java API). embedding_gemma.tflite was exported with a converter newer than what
    // org.tensorflow:tensorflow-lite ever shipped (its op set includes GATHER_ND v5, which
    // that artifact's latest release, 2.17.0, still doesn't support) - LiteRT tracks current
    // op versions, unlike the now-frozen org.tensorflow:tensorflow-lite Maven artifact.
    implementation("com.google.ai.edge.litert:litert:1.4.2")
    // MediaPipe Tasks Text (TextEmbedder) - added alongside the existing raw-Interpreter
    // GemmaEmbedder, not replacing it, to try a longer-context (512 vs 256 token) EmbeddingGemma
    // export packaged as a .task bundle. Fully on-device/offline, no network calls at runtime.
    // 1.0.0 is the current published GA release per Google's Maven metadata (dl.google.com).
    implementation("com.google.mediapipe:tasks-text:1.0.0")
    // Drop-in replacement for android.database.sqlite.* that ships SQLite compiled with runtime
    // loadable-extension support (stock Android's bundled SQLite has this compiled out) - needed
    // to load sqlite-vec (libvec0.so, see app/src/main/jniLibs/) for the vector-store benchmark.
    // Only published via JitPack (see settings.gradle.kts), not Maven Central.
    implementation("com.github.requery:sqlite-android:3.49.0")
}
