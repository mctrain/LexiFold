#include "neural_engine.h"
#include <algorithm>
#include <cstring>
#include <numeric>

GRUEngine::GRUEngine() {
    reset();
}

bool GRUEngine::loadWeights(const uint8_t* data, size_t size) {
    // Minimum header: magic(4) + version(4) + embed_dim(4) + hidden_dim(4) + vocab(4) = 20 bytes
    if (size < 20) return false;

    // Check magic
    uint32_t magic;
    std::memcpy(&magic, data, 4);
    if (magic != NN_MAGIC) return false;

    // Read header
    uint32_t version, ed, hd, vs;
    std::memcpy(&version, data + 4, 4);
    std::memcpy(&ed, data + 8, 4);
    std::memcpy(&hd, data + 12, 4);
    std::memcpy(&vs, data + 16, 4);

    if (version != NN_VERSION || vs != NN_VOCAB) return false;

    embed_dim_ = static_cast<int>(ed);
    hidden_dim_ = static_cast<int>(hd);
    int input_size = embed_dim_ + hidden_dim_;

    // Calculate expected size
    size_t expected = 20; // header
    expected += NN_VOCAB * embed_dim_ * 4;        // embed
    expected += input_size * hidden_dim_ * 4;     // W_z
    expected += hidden_dim_ * 4;                  // b_z
    expected += input_size * hidden_dim_ * 4;     // W_r
    expected += hidden_dim_ * 4;                  // b_r
    expected += input_size * hidden_dim_ * 4;     // W_h
    expected += hidden_dim_ * 4;                  // b_h
    expected += hidden_dim_ * NN_VOCAB * 4;       // out_w
    expected += NN_VOCAB * 4;                     // out_b

    if (size < expected) return false;

    // Helper to read float array
    const float* ptr = reinterpret_cast<const float*>(data + 20);
    auto readVec = [&](std::vector<float>& v, int count) {
        v.resize(count);
        std::memcpy(v.data(), ptr, count * sizeof(float));
        ptr += count;
    };

    readVec(embed_, NN_VOCAB * embed_dim_);
    readVec(W_z_, input_size * hidden_dim_);
    readVec(b_z_, hidden_dim_);
    readVec(W_r_, input_size * hidden_dim_);
    readVec(b_r_, hidden_dim_);
    readVec(W_h_, input_size * hidden_dim_);
    readVec(b_h_, hidden_dim_);
    readVec(out_w_, hidden_dim_ * NN_VOCAB);
    readVec(out_b_, NN_VOCAB);

    hidden_.resize(hidden_dim_, 0.0f);
    loaded_ = true;

    // Initialize with uniform frequencies
    freq_table_.fill(1);
    freq_total_ = NN_VOCAB;

    return true;
}

void GRUEngine::reset() {
    if (hidden_dim_ > 0) {
        std::fill(hidden_.begin(), hidden_.end(), 0.0f);
    }
    freq_table_.fill(1);
    freq_total_ = NN_VOCAB;
}

float GRUEngine::sigmoidf(float x) {
    x = std::max(-15.0f, std::min(15.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

float GRUEngine::tanhf_safe(float x) {
    x = std::max(-15.0f, std::min(15.0f, x));
    return std::tanh(x);
}

void GRUEngine::forward(int byte_val) {
    if (!loaded_ || byte_val < 0 || byte_val >= NN_VOCAB) return;

    int input_size = embed_dim_ + hidden_dim_;

    // Embedding lookup
    const float* x = &embed_[byte_val * embed_dim_];

    // Concatenate [x, h]
    std::vector<float> xh(input_size);
    std::memcpy(xh.data(), x, embed_dim_ * sizeof(float));
    std::memcpy(xh.data() + embed_dim_, hidden_.data(), hidden_dim_ * sizeof(float));

    // Update gate: z = sigmoid(xh @ W_z + b_z)
    std::vector<float> z(hidden_dim_);
    for (int j = 0; j < hidden_dim_; j++) {
        float sum = b_z_[j];
        for (int i = 0; i < input_size; i++) {
            sum += xh[i] * W_z_[i * hidden_dim_ + j];
        }
        z[j] = sigmoidf(sum);
    }

    // Reset gate: r = sigmoid(xh @ W_r + b_r)
    std::vector<float> r(hidden_dim_);
    for (int j = 0; j < hidden_dim_; j++) {
        float sum = b_r_[j];
        for (int i = 0; i < input_size; i++) {
            sum += xh[i] * W_r_[i * hidden_dim_ + j];
        }
        r[j] = sigmoidf(sum);
    }

    // Candidate: h_cand = tanh([x, r*h] @ W_h + b_h)
    std::vector<float> xrh(input_size);
    std::memcpy(xrh.data(), x, embed_dim_ * sizeof(float));
    for (int i = 0; i < hidden_dim_; i++) {
        xrh[embed_dim_ + i] = r[i] * hidden_[i];
    }

    std::vector<float> h_cand(hidden_dim_);
    for (int j = 0; j < hidden_dim_; j++) {
        float sum = b_h_[j];
        for (int i = 0; i < input_size; i++) {
            sum += xrh[i] * W_h_[i * hidden_dim_ + j];
        }
        h_cand[j] = tanhf_safe(sum);
    }

    // New hidden state: h = (1-z)*h + z*h_cand
    for (int i = 0; i < hidden_dim_; i++) {
        hidden_[i] = (1.0f - z[i]) * hidden_[i] + z[i] * h_cand[i];
    }

    // Output logits: logits = h @ out_w + out_b
    std::vector<float> logits(NN_VOCAB);
    for (int j = 0; j < NN_VOCAB; j++) {
        float sum = out_b_[j];
        for (int i = 0; i < hidden_dim_; i++) {
            sum += hidden_[i] * out_w_[i * NN_VOCAB + j];
        }
        logits[j] = sum;
    }

    logitsToFrequencies(logits);
}

void GRUEngine::logitsToFrequencies(const std::vector<float>& logits) {
    // Softmax
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(NN_VOCAB);
    float sumExp = 0.0f;
    for (int i = 0; i < NN_VOCAB; i++) {
        probs[i] = std::exp(logits[i] - maxLogit);
        sumExp += probs[i];
    }

    // Quantize to integer frequencies with target total
    int target_total = NN_MAX_TOTAL / 2; // use half of max for headroom
    freq_total_ = 0;
    for (int i = 0; i < NN_VOCAB; i++) {
        int count = static_cast<int>(probs[i] / sumExp * target_total);
        if (count < 1) count = 1;
        freq_table_[i] = count;
        freq_total_ += count;
    }
}

void GRUEngine::getFrequency(int symbol, int& cumLow, int& cumHigh) const {
    cumLow = 0;
    for (int i = 0; i < symbol; i++) {
        cumLow += freq_table_[i];
    }
    cumHigh = cumLow + freq_table_[symbol];
}

int GRUEngine::getTotal() const {
    return freq_total_;
}

void GRUEngine::findSymbol(int scaledValue, int& symbol, int& cumLow, int& cumHigh) const {
    int cum = 0;
    for (int i = 0; i < NN_VOCAB; i++) {
        int next = cum + freq_table_[i];
        if (next > scaledValue) {
            symbol = i;
            cumLow = cum;
            cumHigh = next;
            return;
        }
        cum = next;
    }
    symbol = NN_VOCAB - 1;
    cumLow = freq_total_ - freq_table_[NN_VOCAB - 1];
    cumHigh = freq_total_;
}

void GRUEngine::update(int symbol) {
    forward(symbol);
}
