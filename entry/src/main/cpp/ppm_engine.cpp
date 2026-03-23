#include "ppm_engine.h"
#include <algorithm>

PPMEngine::PPMEngine() {
    reset();
}

void PPMEngine::reset() {
    // Order-0: Laplace smoothing
    o0_counts_.fill(1);
    o0_total_ = VOCAB_SIZE;

    // Order-1: Laplace smoothing for all contexts
    o1_counts_.fill(1);
    o1_totals_.fill(VOCAB_SIZE);

    // Order-2/3: empty (lazy allocation)
    o2_contexts_.clear();
    o3_contexts_.clear();

    prev1_ = 0;
    prev2_ = 0;
    prev3_ = 0;

    // Initial blended = order-0
    for (int i = 0; i < VOCAB_SIZE; i++) {
        blended_[i] = 1;
    }
    blended_total_ = VOCAB_SIZE;
}

void PPMEngine::recomputeBlended() {
    // Blend predictions from all available context orders.
    // Higher-order contexts get more weight when they have sufficient data.
    // Pure integer arithmetic: multiply counts by weight, accumulate, normalize.

    // Weights (scaled by 256 to avoid fractions):
    // o0: always weight 1 (baseline)
    // o1: weight based on data in context
    // o2: higher weight when context is established
    // o3: highest weight when context is well-established

    int o1_base = prev1_ * VOCAB_SIZE;
    int o1_t = o1_totals_[prev1_];

    int o2_key = prev2_ * VOCAB_SIZE + prev1_;
    ContextEntry* o2 = nullptr;
    auto it2 = o2_contexts_.find(o2_key);
    if (it2 != o2_contexts_.end()) {
        o2 = &it2->second;
    }

    int o3_key = prev3_ * 65536 + prev2_ * 256 + prev1_;
    ContextEntry* o3 = nullptr;
    auto it3 = o3_contexts_.find(o3_key);
    if (it3 != o3_contexts_.end()) {
        o3 = &it3->second;
    }

    // Dynamic weights: higher-order gets more weight when it has more data
    // w0 = 1 (always)
    // w1 = min(o1_t / 4, 16)  — up to 16x when context seen 64+ times
    // w2 = min(o2.total / 2, 32) — up to 32x when context seen 64+ times
    // w3 = min(o3.total / 1, 64) — up to 64x when context seen 64+ times
    int w0 = 1;
    int w1 = std::min(o1_t / 4, 16);
    if (w1 < 1) w1 = 1;

    int w2 = 0;
    if (o2 && o2->total > 0) {
        w2 = std::min(o2->total / 2, 32);
        if (w2 < 1) w2 = 1;
    }

    int w3 = 0;
    if (o3 && o3->total > 0) {
        w3 = std::min(o3->total, 64);
        if (w3 < 1) w3 = 1;
    }

    int wsum = w0 + w1 + w2 + w3;

    // Compute blended counts using integer scaling
    // For each symbol: blended[s] = sum(wk * countk[s] * SCALE / totalk) / wsum
    // Use SCALE = 4096 for precision
    constexpr int SCALE = 4096;

    blended_total_ = 0;
    for (int s = 0; s < VOCAB_SIZE; s++) {
        int score = 0;

        // Order-0 contribution
        score += w0 * static_cast<int>(o0_counts_[s]) * SCALE / o0_total_;

        // Order-1 contribution
        if (o1_t > 0) {
            score += w1 * static_cast<int>(o1_counts_[o1_base + s]) * SCALE / o1_t;
        }

        // Order-2 contribution
        if (o2 && o2->total > 0) {
            score += w2 * static_cast<int>(o2->counts[s]) * SCALE / o2->total;
        }

        // Order-3 contribution
        if (o3 && o3->total > 0) {
            score += w3 * static_cast<int>(o3->counts[s]) * SCALE / o3->total;
        }

        // Normalize by wsum, ensure minimum 1
        int count = score / wsum;
        if (count < 1) count = 1;
        blended_[s] = count;
        blended_total_ += count;
    }

    // Rescale if total too large for arithmetic coder
    if (blended_total_ > MAX_TOTAL) {
        blended_total_ = 0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            blended_[i] = std::max(1, blended_[i] / 2);
            blended_total_ += blended_[i];
        }
    }
}

void PPMEngine::getFrequency(int symbol, int& cumLow, int& cumHigh) const {
    cumLow = 0;
    for (int i = 0; i < symbol; i++) {
        cumLow += blended_[i];
    }
    cumHigh = cumLow + blended_[symbol];
}

int PPMEngine::getTotal() const {
    return blended_total_;
}

void PPMEngine::findSymbol(int scaledValue, int& symbol, int& cumLow, int& cumHigh) const {
    int cum = 0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        int next = cum + blended_[i];
        if (next > scaledValue) {
            symbol = i;
            cumLow = cum;
            cumHigh = next;
            return;
        }
        cum = next;
    }
    // Fallback
    symbol = VOCAB_SIZE - 1;
    cumLow = blended_total_ - blended_[VOCAB_SIZE - 1];
    cumHigh = blended_total_;
}

void PPMEngine::update(int symbol) {
    // Update order-0
    o0_counts_[symbol]++;
    o0_total_++;
    if (o0_total_ >= MAX_TOTAL) {
        rescaleArray(o0_counts_, o0_total_);
    }

    // Update order-1
    int o1_base = prev1_ * VOCAB_SIZE;
    o1_counts_[o1_base + symbol]++;
    o1_totals_[prev1_]++;
    if (o1_totals_[prev1_] >= MAX_TOTAL) {
        int t = 0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            o1_counts_[o1_base + i] = std::max<uint16_t>(1, o1_counts_[o1_base + i] / 2);
            t += o1_counts_[o1_base + i];
        }
        o1_totals_[prev1_] = t;
    }

    // Update order-2
    int o2_key = prev2_ * VOCAB_SIZE + prev1_;
    auto& o2ctx = o2_contexts_[o2_key];
    o2ctx.counts[symbol]++;
    o2ctx.total++;
    if (o2ctx.total >= MAX_TOTAL) {
        o2ctx.total = 0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            o2ctx.counts[i] = std::max<uint16_t>(1, o2ctx.counts[i] / 2);
            o2ctx.total += o2ctx.counts[i];
        }
    }

    // Update order-3
    int o3_key = prev3_ * 65536 + prev2_ * 256 + prev1_;
    auto& o3ctx = o3_contexts_[o3_key];
    o3ctx.counts[symbol]++;
    o3ctx.total++;
    if (o3ctx.total >= MAX_TOTAL) {
        o3ctx.total = 0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            o3ctx.counts[i] = std::max<uint16_t>(1, o3ctx.counts[i] / 2);
            o3ctx.total += o3ctx.counts[i];
        }
    }

    // Shift context window
    prev3_ = prev2_;
    prev2_ = prev1_;
    prev1_ = symbol;

    // Recompute blended table for next prediction
    recomputeBlended();
}

void PPMEngine::rescaleArray(std::array<uint16_t, VOCAB_SIZE>& counts, int& total) {
    total = 0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        counts[i] = std::max<uint16_t>(1, counts[i] / 2);
        total += counts[i];
    }
}
