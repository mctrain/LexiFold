#pragma once
#include <cstdint>
#include <unordered_map>
#include <array>

// PPM Order-3 with blended context prediction for LexiFold
// Pure integer arithmetic, deterministic, no floating point.

constexpr int VOCAB_SIZE = 256;
constexpr int MAX_TOTAL = 16384;

struct ContextEntry {
    std::array<uint16_t, VOCAB_SIZE> counts;
    int total;

    ContextEntry() : total(0) {
        counts.fill(0);
    }
};

class PPMEngine {
public:
    PPMEngine();

    void reset();

    // Get cumulative frequency range for a symbol
    void getFrequency(int symbol, int& cumLow, int& cumHigh) const;

    // Get total frequency
    int getTotal() const;

    // Find symbol from scaled value (for decoder)
    void findSymbol(int scaledValue, int& symbol, int& cumLow, int& cumHigh) const;

    // Update model with observed symbol
    void update(int symbol);

private:
    // Order-0: global byte frequencies (always available)
    std::array<uint16_t, VOCAB_SIZE> o0_counts_;
    int o0_total_;

    // Order-1: 256 contexts indexed by previous byte
    std::array<uint16_t, VOCAB_SIZE * VOCAB_SIZE> o1_counts_;
    std::array<int, VOCAB_SIZE> o1_totals_;

    // Order-2: hash map, key = prev2*256 + prev1
    std::unordered_map<int, ContextEntry> o2_contexts_;

    // Order-3: hash map, key = prev3*65536 + prev2*256 + prev1
    std::unordered_map<int, ContextEntry> o3_contexts_;

    // Context window
    int prev1_, prev2_, prev3_;

    // Pre-computed blended frequency table (recomputed after each update)
    std::array<int, VOCAB_SIZE> blended_;
    int blended_total_;

    // Recompute blended table from all context layers
    void recomputeBlended();

    // Rescale a count array
    static void rescale(uint16_t* counts, int size, int& total);
    static void rescaleArray(std::array<uint16_t, VOCAB_SIZE>& counts, int& total);
};
