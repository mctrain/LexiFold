#include <napi/native_api.h>
#include "ppm_engine.h"
#include "neural_engine.h"
#include "rwkv_engine.h"

// Global engine instances
static PPMEngine g_engine;
static GRUEngine g_neural;
static RWKVEngine g_rwkv;

static napi_value PpmReset(napi_env env, napi_callback_info info) {
    g_engine.reset();
    return nullptr;
}

static napi_value PpmGetFrequency(napi_env env, napi_callback_info info) {
    // Get symbol argument
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    double symD = 0;
    napi_get_value_double(env, argv[0], &symD);
    int symbol = static_cast<int>(symD);

    int cumLow = 0, cumHigh = 0;
    g_engine.getFrequency(symbol, cumLow, cumHigh);

    // Return [cumLow, cumHigh] as array
    napi_value result;
    napi_create_array_with_length(env, 2, &result);

    napi_value v0, v1;
    napi_create_double(env, static_cast<double>(cumLow), &v0);
    napi_create_double(env, static_cast<double>(cumHigh), &v1);
    napi_set_element(env, result, 0, v0);
    napi_set_element(env, result, 1, v1);

    return result;
}

static napi_value PpmGetTotal(napi_env env, napi_callback_info info) {
    napi_value result;
    napi_create_double(env, static_cast<double>(g_engine.getTotal()), &result);
    return result;
}

static napi_value PpmFindSymbol(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    double svD = 0;
    napi_get_value_double(env, argv[0], &svD);
    int scaledValue = static_cast<int>(svD);

    int symbol = 0, cumLow = 0, cumHigh = 0;
    g_engine.findSymbol(scaledValue, symbol, cumLow, cumHigh);

    // Return [symbol, cumLow, cumHigh] as array
    napi_value result;
    napi_create_array_with_length(env, 3, &result);

    napi_value v0, v1, v2;
    napi_create_double(env, static_cast<double>(symbol), &v0);
    napi_create_double(env, static_cast<double>(cumLow), &v1);
    napi_create_double(env, static_cast<double>(cumHigh), &v2);
    napi_set_element(env, result, 0, v0);
    napi_set_element(env, result, 1, v1);
    napi_set_element(env, result, 2, v2);

    return result;
}

static napi_value PpmUpdate(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    double symD = 0;
    napi_get_value_double(env, argv[0], &symD);
    int symbol = static_cast<int>(symD);

    g_engine.update(symbol);
    return nullptr;
}

// === Neural GRU Backend ===

static napi_value NnLoadModel(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    // Receive ArrayBuffer with weight data
    void* data = nullptr;
    size_t length = 0;
    napi_get_arraybuffer_info(env, argv[0], &data, &length);

    bool ok = g_neural.loadWeights(static_cast<const uint8_t*>(data), length);

    napi_value result;
    napi_get_boolean(env, ok, &result);
    return result;
}

static napi_value NnReset(napi_env env, napi_callback_info info) {
    g_neural.reset();
    return nullptr;
}

static napi_value NnGetFrequency(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    double symD = 0;
    napi_get_value_double(env, argv[0], &symD);

    int cumLow = 0, cumHigh = 0;
    g_neural.getFrequency(static_cast<int>(symD), cumLow, cumHigh);

    napi_value result;
    napi_create_array_with_length(env, 2, &result);
    napi_value v0, v1;
    napi_create_double(env, static_cast<double>(cumLow), &v0);
    napi_create_double(env, static_cast<double>(cumHigh), &v1);
    napi_set_element(env, result, 0, v0);
    napi_set_element(env, result, 1, v1);
    return result;
}

static napi_value NnGetTotal(napi_env env, napi_callback_info info) {
    napi_value result;
    napi_create_double(env, static_cast<double>(g_neural.getTotal()), &result);
    return result;
}

static napi_value NnFindSymbol(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    double svD = 0;
    napi_get_value_double(env, argv[0], &svD);

    int symbol = 0, cumLow = 0, cumHigh = 0;
    g_neural.findSymbol(static_cast<int>(svD), symbol, cumLow, cumHigh);

    napi_value result;
    napi_create_array_with_length(env, 3, &result);
    napi_value v0, v1, v2;
    napi_create_double(env, static_cast<double>(symbol), &v0);
    napi_create_double(env, static_cast<double>(cumLow), &v1);
    napi_create_double(env, static_cast<double>(cumHigh), &v2);
    napi_set_element(env, result, 0, v0);
    napi_set_element(env, result, 1, v1);
    napi_set_element(env, result, 2, v2);
    return result;
}

static napi_value NnUpdate(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    double symD = 0;
    napi_get_value_double(env, argv[0], &symD);
    g_neural.update(static_cast<int>(symD));
    return nullptr;
}

static napi_value NnIsLoaded(napi_env env, napi_callback_info info) {
    napi_value result;
    napi_get_boolean(env, g_neural.isLoaded(), &result);
    return result;
}

// Module initialization
// === RWKV Backend ===

static napi_value RwLoadModel(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    void* data = nullptr; size_t length = 0;
    napi_get_arraybuffer_info(env, argv[0], &data, &length);
    bool ok = g_rwkv.loadWeights(static_cast<const uint8_t*>(data), length);
    napi_value result; napi_get_boolean(env, ok, &result);
    return result;
}

static napi_value RwReset(napi_env env, napi_callback_info info) { g_rwkv.reset(); return nullptr; }

static napi_value RwGetFrequency(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    double d = 0; napi_get_value_double(env, argv[0], &d);
    int cumLow = 0, cumHigh = 0;
    g_rwkv.getFrequency((int)d, cumLow, cumHigh);
    napi_value result; napi_create_array_with_length(env, 2, &result);
    napi_value v0, v1;
    napi_create_double(env, (double)cumLow, &v0); napi_create_double(env, (double)cumHigh, &v1);
    napi_set_element(env, result, 0, v0); napi_set_element(env, result, 1, v1);
    return result;
}

static napi_value RwGetTotal(napi_env env, napi_callback_info info) {
    napi_value r; napi_create_double(env, (double)g_rwkv.getTotal(), &r); return r;
}

static napi_value RwFindSymbol(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    double d = 0; napi_get_value_double(env, argv[0], &d);
    int sym = 0, cumLow = 0, cumHigh = 0;
    g_rwkv.findSymbol((int)d, sym, cumLow, cumHigh);
    napi_value result; napi_create_array_with_length(env, 3, &result);
    napi_value v0, v1, v2;
    napi_create_double(env, (double)sym, &v0); napi_create_double(env, (double)cumLow, &v1); napi_create_double(env, (double)cumHigh, &v2);
    napi_set_element(env, result, 0, v0); napi_set_element(env, result, 1, v1); napi_set_element(env, result, 2, v2);
    return result;
}

static napi_value RwUpdate(napi_env env, napi_callback_info info) {
    size_t argc = 1; napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    double d = 0; napi_get_value_double(env, argv[0], &d);
    g_rwkv.update((int)d); return nullptr;
}

static napi_value RwIsLoaded(napi_env env, napi_callback_info info) {
    napi_value r; napi_get_boolean(env, g_rwkv.isLoaded(), &r); return r;
}

static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        // PPM backend
        {"ppmReset", nullptr, PpmReset, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"ppmGetFrequency", nullptr, PpmGetFrequency, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"ppmGetTotal", nullptr, PpmGetTotal, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"ppmFindSymbol", nullptr, PpmFindSymbol, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"ppmUpdate", nullptr, PpmUpdate, nullptr, nullptr, nullptr, napi_default, nullptr},
        // Neural GRU backend
        {"nnLoadModel", nullptr, NnLoadModel, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"nnReset", nullptr, NnReset, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"nnGetFrequency", nullptr, NnGetFrequency, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"nnGetTotal", nullptr, NnGetTotal, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"nnFindSymbol", nullptr, NnFindSymbol, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"nnUpdate", nullptr, NnUpdate, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"nnIsLoaded", nullptr, NnIsLoaded, nullptr, nullptr, nullptr, napi_default, nullptr},
        // RWKV backend
        {"rwLoadModel", nullptr, RwLoadModel, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"rwReset", nullptr, RwReset, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"rwGetFrequency", nullptr, RwGetFrequency, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"rwGetTotal", nullptr, RwGetTotal, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"rwFindSymbol", nullptr, RwFindSymbol, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"rwUpdate", nullptr, RwUpdate, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"rwIsLoaded", nullptr, RwIsLoaded, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}

static napi_module lexifold_module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "lexifold_native",
    .nm_priv = nullptr,
    .reserved = {0},
};

extern "C" __attribute__((constructor))
void RegisterLexifoldNativeModule(void) {
    napi_module_register(&lexifold_module);
}
