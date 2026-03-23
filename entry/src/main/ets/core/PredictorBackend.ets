// Predictor backends for LexiFold
// Each backend maintains a frequency model used by the arithmetic coder.
// Encoder and decoder must use identical backend + update sequence.

import {
  PredictorBackend, FrequencyInfo, SymbolInfo,
  BYTE_VOCAB_SIZE, MAX_TOTAL_FREQ,
  BACKEND_CPU_REFERENCE, BACKEND_MOCK_STEP, BACKEND_CPU_ORDER1,
  BACKEND_NATIVE_PPM, BACKEND_NEURAL_GRU, BACKEND_RWKV,
  MODEL_ORDER0_ADAPTIVE, MODEL_ORDER1_ADAPTIVE, MODEL_PPM_ORDER3,
  MODEL_GRU_CHAR, MODEL_RWKV_BYTE, MODEL_NONE
} from './Types';
import nativeModule from 'liblexifold_native.so';

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

// === Order-1 Adaptive Backend ===
// 256 contexts (one per previous byte), each with independent frequency counts.
// Significantly better than Order-0 for natural language (~43% vs ~62% on English).

export class Order1AdaptiveBackend implements PredictorBackend {
  readonly id: number = BACKEND_CPU_ORDER1;
  readonly modelId: number = MODEL_ORDER1_ADAPTIVE;
  readonly name: string = 'CPU Order-1 Adaptive';
  readonly vocabSize: number = BYTE_VOCAB_SIZE;

  // Flat array: contexts[prev * 256 + symbol] = count
  private contexts: number[] = [];
  private totals: number[] = [];
  private prev: number = 0;

  constructor() {
    this.reset();
  }

  reset(): void {
    // 256 contexts × 256 symbols = 65536 entries, all initialized to 1
    const size: number = BYTE_VOCAB_SIZE * BYTE_VOCAB_SIZE;
    this.contexts = new Array<number>(size);
    for (let i = 0; i < size; i++) {
      this.contexts[i] = 1;
    }
    this.totals = new Array<number>(BYTE_VOCAB_SIZE);
    for (let i = 0; i < BYTE_VOCAB_SIZE; i++) {
      this.totals[i] = BYTE_VOCAB_SIZE;
    }
    this.prev = 0;
  }

  getFrequency(symbol: number): FrequencyInfo {
    const base: number = this.prev * BYTE_VOCAB_SIZE;
    let cumLow: number = 0;
    for (let i = 0; i < symbol; i++) {
      cumLow += this.contexts[base + i];
    }
    const fi: FrequencyInfo = {
      cumLow: cumLow,
      cumHigh: cumLow + this.contexts[base + symbol]
    };
    return fi;
  }

  getTotal(): number {
    return this.totals[this.prev];
  }

  findSymbol(scaledValue: number): SymbolInfo {
    const base: number = this.prev * BYTE_VOCAB_SIZE;
    let cumFreq: number = 0;
    for (let i = 0; i < BYTE_VOCAB_SIZE; i++) {
      const nextCum: number = cumFreq + this.contexts[base + i];
      if (nextCum > scaledValue) {
        const si: SymbolInfo = { symbol: i, cumLow: cumFreq, cumHigh: nextCum };
        return si;
      }
      cumFreq = nextCum;
    }
    const si: SymbolInfo = {
      symbol: BYTE_VOCAB_SIZE - 1,
      cumLow: this.totals[this.prev] - this.contexts[base + BYTE_VOCAB_SIZE - 1],
      cumHigh: this.totals[this.prev]
    };
    return si;
  }

  update(symbol: number): void {
    const base: number = this.prev * BYTE_VOCAB_SIZE;
    this.contexts[base + symbol]++;
    this.totals[this.prev]++;
    if (this.totals[this.prev] >= MAX_TOTAL_FREQ) {
      this.rescaleContext(this.prev);
    }
    this.prev = symbol;
  }

  private rescaleContext(ctx: number): void {
    const base: number = ctx * BYTE_VOCAB_SIZE;
    this.totals[ctx] = 0;
    for (let i = 0; i < BYTE_VOCAB_SIZE; i++) {
      this.contexts[base + i] = Math.max(1, Math.floor(this.contexts[base + i] / 2));
      this.totals[ctx] += this.contexts[base + i];
    }
  }
}

// === Native PPM Backend ===
// Bridges to C++ PPM Order-3 engine via N-API for high compression ratios.

export class NativePPMBackend implements PredictorBackend {
  readonly id: number = BACKEND_NATIVE_PPM;
  readonly modelId: number = MODEL_PPM_ORDER3;
  readonly name: string = 'PPM Order-3 (Native C++)';
  readonly vocabSize: number = BYTE_VOCAB_SIZE;

  constructor() {
    this.reset();
  }

  reset(): void {
    nativeModule.ppmReset();
  }

  getFrequency(symbol: number): FrequencyInfo {
    const arr: number[] = nativeModule.ppmGetFrequency(symbol);
    const fi: FrequencyInfo = { cumLow: arr[0], cumHigh: arr[1] };
    return fi;
  }

  getTotal(): number {
    return nativeModule.ppmGetTotal();
  }

  findSymbol(scaledValue: number): SymbolInfo {
    const arr: number[] = nativeModule.ppmFindSymbol(scaledValue);
    const si: SymbolInfo = { symbol: arr[0], cumLow: arr[1], cumHigh: arr[2] };
    return si;
  }

  update(symbol: number): void {
    nativeModule.ppmUpdate(symbol);
  }
}

// === Native Neural GRU Backend ===
// Character-level GRU neural network inference via C++ N-API.
// Requires model weights loaded via loadModel() before use.

export class NativeNeuralBackend implements PredictorBackend {
  readonly id: number = BACKEND_NEURAL_GRU;
  readonly modelId: number = MODEL_GRU_CHAR;
  readonly name: string = 'GRU Neural (Native C++)';
  readonly vocabSize: number = BYTE_VOCAB_SIZE;
  private modelLoaded: boolean = false;

  constructor() {
    this.modelLoaded = nativeModule.nnIsLoaded();
  }

  loadModel(weightsBuffer: ArrayBuffer): boolean {
    this.modelLoaded = nativeModule.nnLoadModel(weightsBuffer);
    return this.modelLoaded;
  }

  isModelLoaded(): boolean {
    return this.modelLoaded;
  }

  reset(): void {
    nativeModule.nnReset();
  }

  getFrequency(symbol: number): FrequencyInfo {
    const arr: number[] = nativeModule.nnGetFrequency(symbol);
    const fi: FrequencyInfo = { cumLow: arr[0], cumHigh: arr[1] };
    return fi;
  }

  getTotal(): number {
    return nativeModule.nnGetTotal();
  }

  findSymbol(scaledValue: number): SymbolInfo {
    const arr: number[] = nativeModule.nnFindSymbol(scaledValue);
    const si: SymbolInfo = { symbol: arr[0], cumLow: arr[1], cumHigh: arr[2] };
    return si;
  }

  update(symbol: number): void {
    nativeModule.nnUpdate(symbol);
  }
}

// === Native RWKV Backend ===

export class NativeRWKVBackend implements PredictorBackend {
  readonly id: number = BACKEND_RWKV;
  readonly modelId: number = MODEL_RWKV_BYTE;
  readonly name: string = 'RWKV v4 (Native C++)';
  readonly vocabSize: number = BYTE_VOCAB_SIZE;
  private modelLoaded: boolean = false;

  constructor() {
    this.modelLoaded = nativeModule.rwIsLoaded();
  }

  loadModel(weightsBuffer: ArrayBuffer): boolean {
    this.modelLoaded = nativeModule.rwLoadModel(weightsBuffer);
    return this.modelLoaded;
  }

  isModelLoaded(): boolean {
    return this.modelLoaded;
  }

  reset(): void { nativeModule.rwReset(); }

  getFrequency(symbol: number): FrequencyInfo {
    const arr: number[] = nativeModule.rwGetFrequency(symbol);
    const fi: FrequencyInfo = { cumLow: arr[0], cumHigh: arr[1] };
    return fi;
  }

  getTotal(): number { return nativeModule.rwGetTotal(); }

  findSymbol(scaledValue: number): SymbolInfo {
    const arr: number[] = nativeModule.rwFindSymbol(scaledValue);
    const si: SymbolInfo = { symbol: arr[0], cumLow: arr[1], cumHigh: arr[2] };
    return si;
  }

  update(symbol: number): void { nativeModule.rwUpdate(symbol); }
}

// === Backend Factory ===

export function createBackend(backendId: number): PredictorBackend {
  if (backendId === BACKEND_CPU_REFERENCE) {
    return new CpuReferenceBackend();
  } else if (backendId === BACKEND_CPU_ORDER1) {
    return new Order1AdaptiveBackend();
  } else if (backendId === BACKEND_NATIVE_PPM) {
    return new NativePPMBackend();
  } else if (backendId === BACKEND_NEURAL_GRU) {
    return new NativeNeuralBackend();
  } else if (backendId === BACKEND_RWKV) {
    return new NativeRWKVBackend();
  } else if (backendId === BACKEND_MOCK_STEP) {
    return new MockStepModelBackend();
  }
  // Default fallback
  return new CpuReferenceBackend();
}
