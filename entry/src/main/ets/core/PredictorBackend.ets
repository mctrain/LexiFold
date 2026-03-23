// Predictor backends for LexiFold
// Each backend maintains a frequency model used by the arithmetic coder.
// Encoder and decoder must use identical backend + update sequence.

import {
  PredictorBackend, FrequencyInfo, SymbolInfo,
  BYTE_VOCAB_SIZE, MAX_TOTAL_FREQ,
  BACKEND_CPU_REFERENCE, BACKEND_MOCK_STEP,
  MODEL_ORDER0_ADAPTIVE, MODEL_NONE
} from './Types';

// === CPU Reference Backend: Order-0 Adaptive Frequency Model ===
// Tracks per-symbol counts, Laplace-smoothed (initial count=1 each).
// Rescales when total exceeds MAX_TOTAL_FREQ to keep arithmetic coder precision safe.

export class CpuReferenceBackend implements PredictorBackend {
  readonly id: number = BACKEND_CPU_REFERENCE;
  readonly modelId: number = MODEL_ORDER0_ADAPTIVE;
  readonly name: string = 'CPU Reference (Order-0 Adaptive)';
  readonly vocabSize: number = BYTE_VOCAB_SIZE;

  private counts: number[] = [];
  private total: number = 0;

  constructor() {
    this.reset();
  }

  reset(): void {
    this.counts = new Array<number>(BYTE_VOCAB_SIZE);
    for (let i = 0; i < BYTE_VOCAB_SIZE; i++) {
      this.counts[i] = 1;
    }
    this.total = BYTE_VOCAB_SIZE;
  }

  getFrequency(symbol: number): FrequencyInfo {
    let cumLow: number = 0;
    for (let i = 0; i < symbol; i++) {
      cumLow += this.counts[i];
    }
    return {
      cumLow: cumLow,
      cumHigh: cumLow + this.counts[symbol]
    };
  }

  getTotal(): number {
    return this.total;
  }

  findSymbol(scaledValue: number): SymbolInfo {
    let cumFreq: number = 0;
    for (let i = 0; i < BYTE_VOCAB_SIZE; i++) {
      const nextCum: number = cumFreq + this.counts[i];
      if (nextCum > scaledValue) {
        return {
          symbol: i,
          cumLow: cumFreq,
          cumHigh: nextCum
        };
      }
      cumFreq = nextCum;
    }
    // Fallback: last symbol (should not normally reach here)
    return {
      symbol: BYTE_VOCAB_SIZE - 1,
      cumLow: this.total - this.counts[BYTE_VOCAB_SIZE - 1],
      cumHigh: this.total
    };
  }

  update(symbol: number): void {
    this.counts[symbol]++;
    this.total++;
    if (this.total >= MAX_TOTAL_FREQ) {
      this.rescale();
    }
  }

  private rescale(): void {
    this.total = 0;
    for (let i = 0; i < BYTE_VOCAB_SIZE; i++) {
      this.counts[i] = Math.max(1, Math.floor(this.counts[i] / 2));
      this.total += this.counts[i];
    }
  }
}

// === Mock Step Model Backend ===
// Uniform distribution — no compression benefit, used for interface testing only.

export class MockStepModelBackend implements PredictorBackend {
  readonly id: number = BACKEND_MOCK_STEP;
  readonly modelId: number = MODEL_NONE;
  readonly name: string = 'Mock Step Model (Uniform)';
  readonly vocabSize: number = BYTE_VOCAB_SIZE;

  reset(): void {
    // No state
  }

  getFrequency(symbol: number): FrequencyInfo {
    return {
      cumLow: symbol,
      cumHigh: symbol + 1
    };
  }

  getTotal(): number {
    return BYTE_VOCAB_SIZE;
  }

  findSymbol(scaledValue: number): SymbolInfo {
    const sym: number = Math.min(scaledValue, BYTE_VOCAB_SIZE - 1);
    return {
      symbol: sym,
      cumLow: sym,
      cumHigh: sym + 1
    };
  }

  update(symbol: number): void {
    // No state update — uniform distribution is stateless
  }
}

// === Backend Factory ===

export function createBackend(backendId: number): PredictorBackend {
  if (backendId === BACKEND_CPU_REFERENCE) {
    return new CpuReferenceBackend();
  } else if (backendId === BACKEND_MOCK_STEP) {
    return new MockStepModelBackend();
  }
  // Default fallback
  return new CpuReferenceBackend();
}
