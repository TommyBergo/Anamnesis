/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo

/** Enumerates the ExecuTorch backend delegates a model can run on. */
enum class BackendType {
    XNNPACK,
    QUALCOMM,
    MEDIATEK,
    VULKAN
}
