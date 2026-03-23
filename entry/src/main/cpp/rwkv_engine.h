#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include <string>
#include <unordered_map>

// RWKV v4 byte-level inference engine for LexiFold.
// 2 layers, hidden=128, FFN=512, vocab=256.

constexpr int RW_VOCAB = 256;
constexpr int RW_MAX_TOTAL = 16384;
constexpr uint32_t RW_MAGIC = 0x57524C4C; // "LXRW" LE

struct RWKVTensor {
    std::vector<int> shape;
    std::vector<float> data;
};

struct RWKVLayerState {
    std::vector<float> state_x;   // previous token for time-mix
    std::vector<float> state_A;   // WKV numerator
    std::vector<float> state_B;   // WKV denominator
    std::vector<float> state_p;   // WKV log-normalization
    std::vector<float> state_ffn; // previous token for channel-mix
};

class RWKVEngine {
public:
    bool loadWeights(const uint8_t* data, size_t size);
    bool isLoaded() const { return loaded_; }
    void reset();
    void getFrequency(int symbol, int& cumLow, int& cumHigh) const;
    int getTotal() const;
    void findSymbol(int scaledValue, int& symbol, int& cumLow, int& cumHigh) const;
    void update(int symbol);

private:
    bool loaded_ = false;
    int hidden_dim_ = 0;
    int ffn_dim_ = 0;
    int num_layers_ = 0;

    std::unordered_map<std::string, RWKVTensor> weights_;
    std::vector<RWKVLayerState> states_;
    std::array<int, RW_VOCAB> freq_table_;
    int freq_total_ = RW_VOCAB;

    void forward(int byte_val);
    void logitsToFrequencies(const std::vector<float>& logits);

    // Helpers
    const float* w(const std::string& name) const;
    void layerNorm(std::vector<float>& x, const std::string& prefix) const;
    static void vecAdd(std::vector<float>& dst, const std::vector<float>& src);
    static void matVecMul(std::vector<float>& out, const float* W, const std::vector<float>& x, int rows, int cols);
    static float sigmoidf(float x);
};
