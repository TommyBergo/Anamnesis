# Anamnesis: On-Device RAG over Clinical Records

Reference Android implementation for **Anamnesis**, a benchmark for patient-level
retrieval-augmented generation (RAG) over personal clinical records, evaluated end-to-end on a
consumer smartphone.

This repository is a fork of Meta's [ExecuTorch LlamaDemo](https://github.com/meta-pytorch/executorch-examples/tree/main/llm/android/LlamaDemo)
example app, extended with a complete on-device RAG pipeline: PDF ingestion and chunking, six
retrieval configurations (lexical, dense, and hybrid), four embedded vector-store backends, and
on-device generation across five quantized LLM families — all running entirely on-device, since
the sensitivity of clinical records makes local execution a requirement rather than an
optimization.

## Contents

- [What's in this repository](#whats-in-this-repository)
- [Building the app](#building-the-app)
- [Setup](#setup)
- [Using the app](#using-the-app)
- [The RAG pipeline](#the-rag-pipeline)
- [Reproducing the paper's experiments](#reproducing-the-papers-experiments)
- [Reproducing the dataset](#reproducing-the-dataset)
- [General instrumentation tests](#general-instrumentation-tests)
- [License](#license)

## What's in this repository

This repository contains the **Android application only** — everything needed to build the app,
install it on a device, and run the exact instrumentation tests that produced the paper's
on-device results. The benchmark corpus and QA datasets are bundled directly in the app's test
assets (`app/src/androidTest/assets/rag_benchmark/`) — nothing external needs to be downloaded to
run the evaluation suite described below.

The scripts that rebuild the corpus and QA datasets from raw, credentialed MIMIC-III data ship
alongside the app in `rebuild_anamnesis/` (summarized under
[Reproducing the dataset](#reproducing-the-dataset)) so the whole paper — application and dataset
pipeline — is one self-contained repository. One thing is *not* included: the raw MIMIC-III data
itself (and the JSON derived from it), since PhysioNet credentialing forbids redistributing it; see
`rebuild_anamnesis/README.md` for exactly what that means for reproducing the dataset from scratch.
The full experimental result files behind the paper's reported results are also not included here — they are
maintained as supplementary material outside this repository.

```text
LlamaDemo/
├── app/
│   ├── src/main/…              the app itself: chat UI, model loading, the RAG pipeline
│   │   └── java/…/rag/         retrieval engines, embedders, vector-store repositories
│   └── src/androidTest/…       instrumentation tests, including the paper's RAG evaluation suite
│       └── assets/rag_benchmark/   the bundled corpus (138 de-identified admission PDFs) + QA sets
├── docs/delegates/             per-backend (XNNPACK/QNN/MediaTek) build instructions (upstream)
├── scripts/                    CI test runner
├── rebuild_anamnesis/          scripts that rebuild the corpus + QA datasets from raw MIMIC-III
└── README.md                   this file
```

## Building the app

By default the app depends on the [ExecuTorch library](https://central.sonatype.com/artifact/org.pytorch/executorch-android)
from Maven Central (`org.pytorch:executorch-android`), which bundles the default kernel libraries
(portable, quantized, optimized), LLM-specific kernels, and the XNNPACK backend. No extra setup is
needed to use the pre-built library.

To build your own ExecuTorch Android library instead, copy the resulting AAR to
`app/libs/executorch.aar` and add `useLocalAar=true` to `gradle.properties`. See
[the ExecuTorch Android extension docs](https://github.com/pytorch/executorch/blob/main/extension/android/README.md)
for how to build that AAR.

ExecuTorch supports four delegates (compute backends); this app has been evaluated with XNNPACK
(CPU-based, used for all paper results). Per-delegate setup instructions:


| Delegate                            | Resource                                                                                                    |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| XNNPACK (CPU-based)                 | [docs/delegates/xnnpack_README.md](docs/delegates/xnnpack_README.md)                                        |
| QNN (Qualcomm AI Accelerators)      | [docs/delegates/qualcomm_README.md](docs/delegates/qualcomm_README.md)                                      |
| MediaTek (MediaTek AI Accelerators) | [docs/delegates/mediatek_README.md](docs/delegates/mediatek_README.md)                                      |
| Vulkan                              | [pytorch/executorch Vulkan docs](https://github.com/pytorch/executorch/blob/main/examples/vulkan/README.md) |

Open Android Studio, choose "Open an existing Android Studio project," and select this directory
(the one containing this README). `minSdk` is 28, `compileSdk`/`targetSdk` are 35.

## Setup

Follow [SDK-quick-setup-guide.md](SDK-quick-setup-guide.md) if you don't already have Java/Android
SDK/NDK set up.

Two of the four vector-store backends compared in the paper
depend on prebuilt native libraries that ship pre-bundled under `app/src/main/jniLibs/arm64-v8a/`
(arm64-v8a only — the reference device's architecture):

- **USearch** (`libusearch_c.so`) — from [unum-cloud/usearch releases](https://github.com/unum-cloud/usearch/releases),
  `usearch_android_arm64_2.26.2.zip`. Linked at build time via `app/src/main/cpp/CMakeLists.txt`
  (a small JNI bridge, `usearch_jni.cpp`, is compiled against it through Gradle's
  `externalNativeBuild`).
- **sqlite-vec** (`libvec0.so`) — from [asg017/sqlite-vec releases](https://github.com/asg017/sqlite-vec/releases),
  loaded at runtime as a SQLite extension (via `io.requery:sqlite-android`, since stock Android
  SQLite is built without extension-loading support).

Both files are already committed in this repository, so a normal build needs nothing extra. If
you're replacing either one (e.g. for a different ABI or a newer release), it must land at
`app/src/main/jniLibs/<abi>/libusearch_c.so` or `app/src/main/jniLibs/<abi>/libvec0.so`
respectively — `UsearchVectorRepository`/`SqliteVecVectorRepository` load them from that path at
runtime.

## Using the app

### Push model and tokenizer files to the device

Before selecting a model in the app, push its `.pte` and tokenizer files to the device:

```sh
adb shell mkdir -p /data/local/tmp/llama
adb push <your_model>.pte /data/local/tmp/llama
adb push <your_tokenizer> /data/local/tmp/llama
```

### Load a model and chat

1. Open the settings screen, pick a model/tokenizer/model type, and tap "Load Model."
2. Optional parameters: **Temperature** (default 0, reloads the model on change), **System
   Prompt**, and **User Prompt** template (advanced — lets you edit the raw prompt including
   special tokens).
3. Once loaded, type a prompt and send it.

```java
mModule = new LlmModule(
            ModelUtils.getModelCategory(mCurrentSettingsFields.getModelType()),
            modelPath, tokenizerPath, temperature, dataPath);
int loadResult = mModule.load();
mModule.generate(prompt, sequence_length, MainActivity.this);
```

Implement `onResult(String result)` (called incrementally as tokens are generated) and
`onStats(String stats)` (a JSON payload — see `extension/llm/stats.h` in ExecuTorch for its
fields) in your callback class.

### RAG mode

Beyond free-form chat, the app has a document-grounded chat mode: ingest one or more PDFs, pick an
embedder in settings, and ask questions answered against the ingested documents rather than the
model's parametric knowledge alone. The retrieval engine itself is fixed in code to
`RrfBm25MediaPipeRagEngine` — the paper's Hybrid configuration (BM25 + dense embedding
fused via Reciprocal Rank Fusion, §4.3 Eq. 1) — so the interactive chat mode reproduces the same
retrieval behavior the paper reports, not a separate ad-hoc blend. See
[The RAG pipeline](#the-rag-pipeline) below for how the engines map to the paper's retrieval configurations, and
[Reproducing the paper's experiments](#reproducing-the-papers-experiments) for running the same
evaluations non-interactively via instrumentation tests.

## The RAG pipeline

`app/src/main/java/com/example/executorchllamademo/rag/` implements six retrieval configurations
across four `RagEngine` classes. TF-IDF, BM25, and Hybrid each have a dedicated engine; the three
dense configurations (EmbeddingGemma, multilingual-e5-small, Granite) all run through the same
engine, `MedicalRagEngine`, and differ only in which `Embedder` implementation is injected into it
— so comparing them **is** comparing `MedicalRagEngine` against itself with a different embedder
each time, not three separate engines:


| Paper configuration   | Engine class                                       | Notes                                                                                                                                 |
| --------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| TF-IDF                | `TfIdfRagEngine`                                   | classic TF-IDF, cosine similarity                                                                                                     |
| BM25                  | `Bm25RagEngine`                                    | Okapi BM25                                                                                                                            |
| EmbeddingGemma        | `MedicalRagEngine` + `EmbeddingGemmaEmbedder`      | Google EmbeddingGemma 300M via MediaPipe, 512-token window (default dense embedder)                                                   |
| multilingual-e5-small | `MedicalRagEngine` + `MultilingualE5SmallEmbedder` | intfloat/multilingual-e5-small, 384-dim, 512-token window                                                                             |
| Granite               | `MedicalRagEngine` + `GraniteEmbedder`             | IBM Granite Embedding 311M multilingual, 512-token window                                                                             |
| Hybrid                | `RrfBm25MediaPipeRagEngine`                        | BM25 + EmbeddingGemma fused via Reciprocal Rank Fusion (§4.3 Eq. 1, c=60) —**the default engine used by the interactive chat mode** |

`HybridRagEngine` (a separate, older semantic+TF-IDF weighted blend, exercised only by
`HybridRagEvaluationTest`) and `MedicalRagEngine`'s androidTest-only variants are additional engines
in the app not reported in the paper's retrieval comparison — the mapping above is the authoritative paper-to-code
mapping.

Vector-store backends compared in the paper (`app/src/main/java/.../rag/repository/`):
`MedicalVectorRepository` (ObjectBox/HNSW), `SqliteVecVectorRepository` (SQLite + sqlite-vec),
`UsearchVectorRepository` (USearch), `BruteForceVectorRepository` (linear-scan baseline) — see
[Setup](#setup) for their native-library provenance.

Generators evaluated in the paper: Qwen3-0.6B/1.7B/4B, Llama-3.2-1B, and Phi-4-mini, all
quantized and exported to ExecuTorch `.pte` format; Qwen3-1.7B is the fixed generator used
alongside every retriever for the end-to-end answer-quality comparison.

## Reproducing the paper's experiments

Every retrieval number comes directly from an instrumentation test, run the same way you'd run any Android instrumentation test.

All RAG evaluation tests live in `app/src/androidTest/java/com/example/executorchllamademo/rag/`.

The instrumentation tests below produce the raw generated answers on-device; judging those answers against the ground truth is done by an
external LLM (GPT-5.6) as a separate, offline pass that is not part of this repository.

### Retrieval — one command per engine

```sh
adb shell am instrument -w -r \
  -e class com.example.executorchllamademo.rag.SimpleJsonRagEvaluationTest#runSimpleJsonRagEvaluation \
  -e skipGeneration true \
  com.example.executorchllamademo.test/androidx.test.runner.AndroidJUnitRunner
```


| Engine                | `-e class`                                               | Extra args                               |
| --------------------- | -------------------------------------------------------- | ---------------------------------------- |
| TF-IDF                | `SimpleJsonRagEvaluationTest#runSimpleJsonRagEvaluation` | —                                       |
| BM25                  | `Bm25RagEvaluationTest#runBm25RagEvaluation`             | —                                       |
| EmbeddingGemma        | `MedicalRagEvaluationTest#runMedicalRagEvaluation`       | `-e embedderBackend mediapipe` (default) |
| Granite               | `MedicalRagEvaluationTest#runMedicalRagEvaluation`       | `-e embedderBackend granite`             |
| multilingual-e5-small | `MedicalRagEvaluationTest#runMedicalRagEvaluation`       | `-e embedderBackend e5_small`            |
| Hybrid (RRF)          | `RrfBm25RagEvaluationTest#runRrfBm25RagEvaluation`       | —                                       |

Each has a `MultiDoc*` counterpart (e.g. `MultiDocBm25RagEvaluationTest`) for the multi-document
split. Pass `-e skipGeneration true` for a retrieval-only run (fast — no reader model needed); omit
it (or pass `false`) to also run generation, in which case push a reader model first (see
[above](#push-model-and-tokenizer-files-to-the-device)) and pass its filenames:
`-e modelFile <name>.pte -e tokenizerFile <name>.json`. `-e maxItems N` caps the dataset to the
first N items — useful for a quick smoke run before committing to a full campaign. Results are
written as JSON to AGP's `additionalTestOutputDir`, auto-copied to the host after the run.

### Chunk-size ablation

Every retrieval test accepts `-e chunkSize N -e overlapChars N` to override the ingestion
chunking parameters (default 500/100 - see [The RAG pipeline](#the-rag-pipeline)). The chunk-size ablation
compares neural retrieval at `chunkSize` ∈ {500, 1000, 1500} chars, `overlapChars` fixed at 100:

```sh
adb shell am instrument -w -r \
  -e class com.example.executorchllamademo.rag.MedicalRagEvaluationTest#runMedicalRagEvaluation \
  -e skipGeneration true -e chunkSize 1000 -e overlapChars 100 \
  com.example.executorchllamademo.test/androidx.test.runner.AndroidJUnitRunner
```

### Per-question-type and per-construction-technique breakdowns

These don't need separate commands - they re-slice the retrieval output. Every
single-document item carries a `question_type` field (the breakdown key), and comes from one
of two dataset files, `QA/single_dataset_rulebased.jsonl` or `QA/single_dataset_llm.jsonl` (the
rule-based-vs-LLM-generated split - run the same retrieval test once per file via
`-e datasetAsset QA/single_dataset_llm.jsonl` and compare against the rule-based default).

### Generator sweep

```sh
adb shell am instrument -w -r \
  -e class com.example.executorchllamademo.rag.GeneratorSweepEvaluationTest#runClosedBookEvaluation \
  com.example.executorchllamademo.test/androidx.test.runner.AndroidJUnitRunner
```

Three conditions live in `GeneratorSweepEvaluationTest`: `runClosedBookEvaluation`,
`runFullContextEvaluation`, `runOracleEvidenceEvaluation`. The fourth (RAG) condition reuses
`Bm25RagEvaluationTest` directly. Swap `-e modelFile`/`-e tokenizerFile` to sweep across the five
generators.

### Vector-store comparison

```sh
adb shell am instrument -w -r \
  -e class com.example.executorchllamademo.rag.VectorStoreBenchmarkTest#runVectorStoreBenchmark \
  com.example.executorchllamademo.test/androidx.test.runner.AndroidJUnitRunner
```

One run benchmarks all four backends together (ObjectBox, sqlite-vec, USearch, brute-force) on the
same ingested corpus.

### Cross-patient probe

The 32 LLM-generated cross-patient items live at
`app/src/androidTest/assets/rag_benchmark/QA/multidoc_dataset_llm_cross_patient.jsonl`. No
dedicated test class exists for this probe, but the existing multi-document tests already retrieve
per-patient and merge per item (keyed by however many distinct `patientId`s an item's `sources`
span, one or more), which is exactly what a cross-patient item needs - confirmed by actually
running it:

```sh
adb shell am instrument -w -r \
  -e class com.example.executorchllamademo.rag.RrfBm25MultiDocRagEvaluationTest#runRrfBm25MultiDocRagEvaluation \
  -e skipGeneration true -e datasetAsset QA/multidoc_dataset_llm_cross_patient.jsonl \
  com.example.executorchllamademo.test/androidx.test.runner.AndroidJUnitRunner
```

Any of the four `MultiDoc*RagEvaluationTest` classes works the same way via `-e datasetAsset`, one
per retriever, matching the reported cross-patient comparison. The matching 35-item rule-based half is not bundled
here - `rebuild_anamnesis/split_qa_multidoc_by_patient.py` produces it (documented in that folder's
own README as "not currently used, produced for completeness") but it isn't shipped as a test
asset in this repository yet.

## Reproducing the dataset

The 138-admission PDF corpus and the rule-based QA datasets bundled in
`app/src/androidTest/assets/rag_benchmark/` were built from raw, credentialed MIMIC-III data by the
pipeline in [`rebuild_anamnesis/`](rebuild_anamnesis/README.md): extraction and stratified sampling
from the raw MIMIC-III tables, PDF rendering, and the deterministic single- and multi-document QA
generators. The scripts themselves need no credentialed access and are ordinary Python; only
*running* them does, since their input (the raw MIMIC-III CSVs and the JSON derived from them) is
governed by the PhysioNet Data Use Agreement and is deliberately not bundled anywhere in this
repository. See `rebuild_anamnesis/README.md` for exact steps and evidence of provenance.

## General instrumentation tests

Beyond the RAG suite, the app includes two general sanity/workflow tests:

1. **SanityCheck** — basic model loading and generation, verifying the LLM module loads a model
   and generates tokens.
2. **UIWorkflowTest** — simulated user interactions: `testModelLoadingWorkflow` (select and load a
   model/tokenizer) and `testSendMessageAndReceiveResponse` (send a message, receive a response).

The test model (`stories110M.pte`) and its tokenizer are **automatically downloaded** by Gradle
before these run. To prepare them manually instead:

```sh
# Install the executorch Python package first:
# https://docs.pytorch.org/executorch/stable/getting-started.html#installation

curl -C - -Ls "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt" --output stories110M.pt
curl -C - -Ls "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model" --output tokenizer.model

touch params.json
echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json

python -m executorch.extension.llm.export.export_llm base.checkpoint=stories110M.pt base.params=params.json \
  model.dtype_override="fp16" export.output_name=stories110M.pte model.use_kv_cache=True

adb shell mkdir -p /data/local/tmp/llama
adb push stories110M.pte /data/local/tmp/llama
adb push tokenizer.model /data/local/tmp/llama
```

Run via Gradle model presets, which handle the download automatically:

```sh
./gradlew connectedCheck -PmodelPreset=stories   # tiny model, quick testing (default)
./gradlew connectedCheck -PmodelPreset=llama     # Llama 3.2 1B
./gradlew connectedCheck -PmodelPreset=qwen3     # Qwen3 4B, INT8/INT4
./gradlew connectedCheck -PmodelPreset=custom -PcustomPteUrl=... -PcustomTokenizerUrl=...
./gradlew connectedCheck -PmodelPreset=stories -PskipModelDownload=true   # reuse files already on device

# Run one specific class:
./gradlew connectedCheck -PmodelPreset=stories \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.executorchllamademo.SanityCheck
```

## License

This app is a fork of Meta's ExecuTorch LlamaDemo example; inherited files remain under the
BSD-style license in `executorch-examples`'s root `LICENSE`. The RAG pipeline, evaluation suite,
and dataset-reproduction scripts added for the Anamnesis paper are MIT-licensed, per the paper's
own statement (footnote 1): the license covers this benchmarking code and the scripts that rebuild
Anamnesis from credentialed MIMIC-III data; the underlying clinical text itself is governed by the
PhysioNet Data Use Agreement and is not redistributed.

## Reporting issues

For issues with the underlying ExecuTorch runtime or delegates, see
[pytorch/executorch](https://github.com/pytorch/executorch/issues/new) or the
[ExecuTorch Discord](https://lnkd.in/gWCM4ViK).
