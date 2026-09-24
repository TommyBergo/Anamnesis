/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
buildscript {
    repositories {
        google()
        mavenCentral() // <-- ObjectBox si trova qui!
    }
    dependencies {
        // Aggiungi il classpath di ObjectBox
        classpath("io.objectbox:objectbox-gradle-plugin:4.0.1")
    }
}
// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.0" apply false
}
