#include "rwkv_engine.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

bool RWKVEngine::loadWeights(const uint8_t* data, size_t size) {
    if (size < 24) return false;

    uint32_t magic;
    std::memcpy(&magic, data, 4);
    if (magic != RW_MAGIC) return false;

    uint32_t version, hd, nl, fd, vs;
    std::memcpy(&version, data + 4, 4);
    std::memcpy(&hd, data + 8, 4);
    std::memcpy(&nl, data + 12, 4);
    std::memcpy(&fd, data + 16, 4);
    std::memcpy(&vs, data + 20, 4);

    hidden_dim_ = (int)hd;
    num_layers_ = (int)nl;
    ffn_dim_ = (int)fd;

    // Read number of tensors
    uint32_t num_tensors;
    std::memcpy(&num_tensors, data + 24, 4);
    size_t off = 28;

    weights_.clear();
    for (uint32_t t = 0; t < num_tensors; t++) {
        // Read key name
        uint32_t key_len;
        std::memcpy(&key_len, data + off, 4); off += 4;
        std::string key((const char*)(data + off), key_len);
        off += key_len;

        // Read shape
        uint32_t ndim;
        std::memcpy(&ndim, data + off, 4); off += 4;
        RWKVTensor tensor;
        int total = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint32_t s;
            std::memcpy(&s, data + off, 4); off += 4;
            tensor.shape.push_back((int)s);
            total *= (int)s;
        }

        tensor.data.resize(total);
        std::memcpy(tensor.data.data(), data + off, total * sizeof(float));
        off += total * sizeof(float);

        weights_[key] = std::move(tensor);
    }

    loaded_ = true;
    reset();
    return true;
}

void RWKVEngine::reset() {
    states_.clear();
    states_.resize(num_layers_);
    for (int i = 0; i < num_layers_; i++) {
        states_[i].state_x.assign(hidden_dim_, 0.0f);
        states_[i].state_A.assign(hidden_dim_, 0.0f);
        states_[i].state_B.assign(hidden_dim_, 0.0f);
        states_[i].state_p.assign(hidden_dim_, -1e30f);
        states_[i].state_ffn.assign(hidden_dim_, 0.0f);
    }
    freq_table_.fill(1);
    freq_total_ = RW_VOCAB;
}

const float* RWKVEngine::w(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) return nullptr;
    return it->second.data.data();
}

float RWKVEngine::sigmoidf(float x) {
    x = std::max(-15.0f, std::min(15.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

void RWKVEngine::layerNorm(std::vector<float>& x, const std::string& prefix) const {
    const float* wt = w(prefix + ".weight");
    const float* bi = w(prefix + ".bias");
    if (!wt || !bi) return;

    int n = (int)x.size();
    float mean = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;

    float var = 0;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / std::sqrt(var + 1e-5f);

    for (int i = 0; i < n; i++) {
        x[i] = (x[i] - mean) * inv * wt[i] + bi[i];
    }
}

void RWKVEngine::vecAdd(std::vector<float>& dst, const std::vector<float>& src) {
    for (size_t i = 0; i < dst.size(); i++) dst[i] += src[i];
}

void RWKVEngine::matVecMul(std::vector<float>& out, const float* W, const std::vector<float>& x, int rows, int cols) {
    out.resize(rows);
    for (int r = 0; r < rows; r++) {
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            sum += W[r * cols + c] * x[c];
        }
        out[r] = sum;
    }
}

void RWKVEngine::forward(int byte_val) {
    if (!loaded_ || byte_val < 0 || byte_val >= RW_VOCAB) return;

    int H = hidden_dim_;
    int F = ffn_dim_;

    // Embedding lookup
    const float* emb_w = w("emb.weight");
    std::vector<float> x(H);
    for (int i = 0; i < H; i++) x[i] = emb_w[byte_val * H + i];

    // ln0
    layerNorm(x, "ln0");

    // Process each block
    for (int layer = 0; layer < num_layers_; layer++) {
        std::string p = "blocks." + std::to_string(layer);
        auto& st = states_[layer];

        // === TimeMix ===
        std::vector<float> x_res = x;
        layerNorm(x, p + ".ln1");

        // Mix with previous token
        const float* mix_k = w(p + ".time_mix.time_mix_k");
        const float* mix_v = w(p + ".time_mix.time_mix_v");
        const float* mix_r = w(p + ".time_mix.time_mix_r");

        std::vector<float> xk(H), xv(H), xr(H);
        for (int i = 0; i < H; i++) {
            xk[i] = x[i] * mix_k[i] + st.state_x[i] * (1.0f - mix_k[i]);
            xv[i] = x[i] * mix_v[i] + st.state_x[i] * (1.0f - mix_v[i]);
            xr[i] = x[i] * mix_r[i] + st.state_x[i] * (1.0f - mix_r[i]);
        }

        // Project k, v, r
        std::vector<float> k, v, r;
        matVecMul(k, w(p + ".time_mix.key.weight"), xk, H, H);
        matVecMul(v, w(p + ".time_mix.value.weight"), xv, H, H);
        matVecMul(r, w(p + ".time_mix.receptance.weight"), xr, H, H);
        for (int i = 0; i < H; i++) r[i] = sigmoidf(r[i]);

        // WKV computation
        const float* time_first = w(p + ".time_mix.time_first");
        const float* time_decay = w(p + ".time_mix.time_decay");

        std::vector<float> wkv(H);
        for (int i = 0; i < H; i++) {
            float ww = time_first[i] + k[i];
            float pp = std::max(st.state_p[i], ww);
            float e1 = std::exp(st.state_p[i] - pp);
            float e2 = std::exp(ww - pp);
            float a = e1 * st.state_A[i] + e2 * v[i];
            float b = e1 * st.state_B[i] + e2;
            wkv[i] = a / (b + 1e-8f);
        }

        // rwkv = r * wkv
        for (int i = 0; i < H; i++) wkv[i] *= r[i];

        // Output projection
        std::vector<float> att_out;
        matVecMul(att_out, w(p + ".time_mix.output.weight"), wkv, H, H);

        // Update WKV state
        for (int i = 0; i < H; i++) {
            float ww2 = st.state_p[i] + time_decay[i];
            float pp2 = std::max(ww2, k[i]);
            float e1 = std::exp(ww2 - pp2);
            float e2 = std::exp(k[i] - pp2);
            st.state_A[i] = e1 * st.state_A[i] + e2 * v[i];
            st.state_B[i] = e1 * st.state_B[i] + e2;
            st.state_p[i] = pp2;
        }
        // Save current x as state for next token's time-mix
        st.state_x = x;  // x before ln1 was x_res, but after ln1 is x. We want pre-LN.
        // Actually the Python code saves x (the input to time_mix, post-LN).
        // Let me follow Python: state_x = x (post-LN, which is what the RWKV uses as "previous")
        // Wait, in Python: new_state_x = x.detach() where x is the post-LN input to time_mix
        // The saved state_x is used in mixing: xk = x * mix_k + state_x * (1 - mix_k)
        // where x is the current post-LN. So state_x should be the previous post-LN.
        // But above I already modified x with layerNorm. So st.state_x = x (post-LN) is correct.

        // Residual
        x = x_res;
        vecAdd(x, att_out);

        // === ChannelMix ===
        x_res = x;
        layerNorm(x, p + ".ln2");

        const float* cm_mix_k = w(p + ".channel_mix.time_mix_k");
        const float* cm_mix_r = w(p + ".channel_mix.time_mix_r");

        std::vector<float> cmk(H), cmr(H);
        for (int i = 0; i < H; i++) {
            cmk[i] = x[i] * cm_mix_k[i] + st.state_ffn[i] * (1.0f - cm_mix_k[i]);
            cmr[i] = x[i] * cm_mix_r[i] + st.state_ffn[i] * (1.0f - cm_mix_r[i]);
        }

        st.state_ffn = x; // save post-LN x as state for next token

        // FFN: key → relu² → value, gated by receptance
        std::vector<float> fk, fv, fr;
        matVecMul(fk, w(p + ".channel_mix.key.weight"), cmk, F, H);
        for (int i = 0; i < F; i++) {
            fk[i] = std::max(0.0f, fk[i]);
            fk[i] = fk[i] * fk[i]; // squared ReLU
        }
        matVecMul(fv, w(p + ".channel_mix.value.weight"), fk, H, F);
        matVecMul(fr, w(p + ".channel_mix.receptance.weight"), cmr, H, H);
        for (int i = 0; i < H; i++) fr[i] = sigmoidf(fr[i]);

        for (int i = 0; i < H; i++) fv[i] *= fr[i];

        x = x_res;
        vecAdd(x, fv);
    }

    // ln_out + head
    layerNorm(x, "ln_out");

    std::vector<float> logits;
    matVecMul(logits, w("head.weight"), x, RW_VOCAB, hidden_dim_);

    logitsToFrequencies(logits);
}

void RWKVEngine::logitsToFrequencies(const std::vector<float>& logits) {
    float maxL = *std::max_element(logits.begin(), logits.end());
    float sumExp = 0;
    std::vector<float> probs(RW_VOCAB);
    for (int i = 0; i < RW_VOCAB; i++) {
        probs[i] = std::exp(logits[i] - maxL);
        sumExp += probs[i];
    }
    int target = RW_MAX_TOTAL / 2;
    freq_total_ = 0;
    for (int i = 0; i < RW_VOCAB; i++) {
        int c = std::max(1, (int)(probs[i] / sumExp * target));
        freq_table_[i] = c;
        freq_total_ += c;
    }
}

void RWKVEngine::getFrequency(int symbol, int& cumLow, int& cumHigh) const {
    cumLow = 0;
    for (int i = 0; i < symbol; i++) cumLow += freq_table_[i];
    cumHigh = cumLow + freq_table_[symbol];
}

int RWKVEngine::getTotal() const { return freq_total_; }

void RWKVEngine::findSymbol(int scaledValue, int& symbol, int& cumLow, int& cumHigh) const {
    int cum = 0;
    for (int i = 0; i < RW_VOCAB; i++) {
        int next = cum + freq_table_[i];
        if (next > scaledValue) {
            symbol = i; cumLow = cum; cumHigh = next; return;
        }
        cum = next;
    }
    symbol = RW_VOCAB - 1;
    cumLow = freq_total_ - freq_table_[RW_VOCAB - 1];
    cumHigh = freq_total_;
}

void RWKVEngine::update(int symbol) { forward(symbol); }
