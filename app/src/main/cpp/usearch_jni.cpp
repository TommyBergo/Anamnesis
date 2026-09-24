// Minimal JNI bridge over USearch's flat C API (usearch.h), just enough to back
// UsearchVectorRepository for the Table 10 vector-store benchmark: init/reserve/add/search/
// size/memory-usage/free. Not a general-purpose USearch binding - only the subset
// VectorStoreBenchmarkTest actually exercises (cosine metric, f32 vectors, single-vector keys).
#include <jni.h>
#include <string>
#include <vector>
#include "usearch.h"

namespace {

void throwIfError(JNIEnv* env, usearch_error_t error) {
    if (error != nullptr) {
        jclass exClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exClass, error);
    }
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_executorchllamademo_rag_repository_UsearchNative_nativeInit(
    JNIEnv* env, jobject /*thiz*/, jint dimensions) {
    usearch_init_options_t options{};
    options.metric_kind = usearch_metric_cos_k;
    options.quantization = usearch_scalar_f32_k;
    options.dimensions = static_cast<size_t>(dimensions);
    options.connectivity = 0;       // 0 = library default
    options.expansion_add = 0;      // 0 = library default
    options.expansion_search = 0;   // 0 = library default
    options.multi = false;

    usearch_error_t error = nullptr;
    usearch_index_t index = usearch_init(&options, &error);
    throwIfError(env, error);
    return reinterpret_cast<jlong>(index);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_executorchllamademo_rag_repository_UsearchNative_nativeReserve(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jlong capacity) {
    usearch_error_t error = nullptr;
    usearch_reserve(reinterpret_cast<usearch_index_t>(handle), static_cast<size_t>(capacity), &error);
    throwIfError(env, error);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_executorchllamademo_rag_repository_UsearchNative_nativeAdd(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jlong key, jfloatArray vector) {
    jfloat* vec = env->GetFloatArrayElements(vector, nullptr);
    usearch_error_t error = nullptr;
    usearch_add(
        reinterpret_cast<usearch_index_t>(handle),
        static_cast<usearch_key_t>(key),
        vec, usearch_scalar_f32_k, &error);
    env->ReleaseFloatArrayElements(vector, vec, JNI_ABORT);
    throwIfError(env, error);
}

// Returns the number of matches found; fills keysOut/distancesOut (must be pre-sized to `count`).
extern "C" JNIEXPORT jint JNICALL
Java_com_example_executorchllamademo_rag_repository_UsearchNative_nativeSearch(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jfloatArray query, jint count,
    jlongArray keysOut, jfloatArray distancesOut) {
    jfloat* q = env->GetFloatArrayElements(query, nullptr);

    std::vector<usearch_key_t> keys(static_cast<size_t>(count));
    std::vector<usearch_distance_t> distances(static_cast<size_t>(count));

    usearch_error_t error = nullptr;
    size_t found = usearch_search(
        reinterpret_cast<usearch_index_t>(handle),
        q, usearch_scalar_f32_k, static_cast<size_t>(count),
        keys.data(), distances.data(), &error);

    env->ReleaseFloatArrayElements(query, q, JNI_ABORT);
    throwIfError(env, error);

    // usearch_key_t is uint64_t; JNI has no unsigned long, reinterpret bit-for-bit into jlong -
    // safe here since our own keys are always small, non-negative insertion indices.
    std::vector<jlong> jkeys(found);
    for (size_t i = 0; i < found; i++) jkeys[i] = static_cast<jlong>(keys[i]);
    env->SetLongArrayRegion(keysOut, 0, static_cast<jsize>(found), jkeys.data());
    env->SetFloatArrayRegion(distancesOut, 0, static_cast<jsize>(found), distances.data());

    return static_cast<jint>(found);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_executorchllamademo_rag_repository_UsearchNative_nativeSize(
    JNIEnv* env, jobject /*thiz*/, jlong handle) {
    usearch_error_t error = nullptr;
    size_t size = usearch_size(reinterpret_cast<usearch_index_t>(handle), &error);
    throwIfError(env, error);
    return static_cast<jlong>(size);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_executorchllamademo_rag_repository_UsearchNative_nativeMemoryUsage(
    JNIEnv* env, jobject /*thiz*/, jlong handle) {
    usearch_error_t error = nullptr;
    size_t bytes = usearch_memory_usage(reinterpret_cast<usearch_index_t>(handle), &error);
    throwIfError(env, error);
    return static_cast<jlong>(bytes);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_executorchllamademo_rag_repository_UsearchNative_nativeSave(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jstring path) {
    const char* pathChars = env->GetStringUTFChars(path, nullptr);
    usearch_error_t error = nullptr;
    usearch_save(reinterpret_cast<usearch_index_t>(handle), pathChars, &error);
    env->ReleaseStringUTFChars(path, pathChars);
    throwIfError(env, error);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_executorchllamademo_rag_repository_UsearchNative_nativeFree(
    JNIEnv* env, jobject /*thiz*/, jlong handle) {
    usearch_error_t error = nullptr;
    usearch_free(reinterpret_cast<usearch_index_t>(handle), &error);
    throwIfError(env, error);
}
