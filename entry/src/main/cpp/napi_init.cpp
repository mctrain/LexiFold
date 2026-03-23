#include <napi/native_api.h>
#include "ppm_engine.h"

// Global PPM engine instance (one per process, reset between chunks)
static PPMEngine g_engine;

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

// Module initialization
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"ppmReset", nullptr, PpmReset, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"ppmGetFrequency", nullptr, PpmGetFrequency, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"ppmGetTotal", nullptr, PpmGetTotal, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"ppmFindSymbol", nullptr, PpmFindSymbol, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"ppmUpdate", nullptr, PpmUpdate, nullptr, nullptr, nullptr, napi_default, nullptr},
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
