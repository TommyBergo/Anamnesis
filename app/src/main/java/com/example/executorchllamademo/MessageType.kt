/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo

/** Enumerates the kind of content a chat message carries: plain text, an image, or a system message. */
enum class MessageType {
    TEXT,
    IMAGE,
    SYSTEM
}
