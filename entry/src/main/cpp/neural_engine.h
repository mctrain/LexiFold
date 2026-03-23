#pragma once
#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>
#include <cmath>

// Tiny character-level GRU inference engine for LexiFold.
// Architecture: Embedding(256→16) → GRU(16→64) → Linear(64→256)
// Weights loaded from binary file at runtime.

constexpr int NN_VOCAB = 256;
constexpr int NN_MAX_TOTAL = 16384;

// Weight file magic and version
constexpr uint32_t NN_MAGIC = 0x4E4E584C; // "LXNN" little-endian
constexpr uint32_t NN_VERSION = 1;

class GRUEngine {
public:
    GRUEngine();

    // Load weights from binary data. Returns true on success.
    bool loadWeights(const uint8_t* data, size_t size);
    bool isLoaded() const { return loaded_; }

    // Reset hidden state (call between compression chunks)
    void reset();

    // Interface matching PredictorBackend
    void getFrequency(int symbol, int& cumLow, int& cumHigh) const;
    int getTotal() const;
    void findSymbol(int scaledValue, int& symbol, int& cumLow, int& cumHigh) const;

    // Update: run one GRU step, update hidden state and frequency table
    void update(int symbol);

private:
    bool loaded_ = false;

    // Model dimensions (from weight file)
    int embed_dim_ = 0;
    int hidden_dim_ = 0;

    // Weights
    std::vector<float> embed_;       // [vocab, embed_dim]
    std::vector<float> W_z_, b_z_;   // update gate
    std::vector<float> W_r_, b_r_;   // reset gate
    std::vector<float> W_h_, b_h_;   // candidate
    std::vector<float> out_w_, out_b_; // output linear

    // State
    std::vector<float> hidden_;      // [hidden_dim]

    // Pre-computed frequency table (updated after each step)
    std::array<int, NN_VOCAB> freq_table_;
    int freq_total_ = NN_VOCAB;

    // Forward pass: compute logits from input byte, update hidden state
    void forward(int byte_val);

    // Convert logits to integer frequency table
    void logitsToFrequencies(const std::vector<float>& logits);

    // Math helpers
    static float sigmoidf(float x);
    static float tanhf_safe(float x);
};
