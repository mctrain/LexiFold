# NPU Integration Guide

## Overview

LexiFold's architecture separates the prediction backend from the encoding pipeline.
To integrate an NPU-accelerated model, implement the `PredictorBackend` interface.

## Required Interface

```typescript
interface PredictorBackend {
  id: string;
  name: string;
  reset(): void;
  getFrequency(symbol: number): { cumLow: number, cumHigh: number };
  getTotal(): number;
  findSymbol(scaledValue: number): { symbol: number, cumLow: number, cumHigh: number };
  update(symbol: number): void;
}
```

## Integration Steps

1. **Model Preparation**
   - Train or obtain a small language model (e.g., RWKV, small transformer)
   - Export to MindSpore Lite / ONNX format compatible with HiAI
   - The model must implement: `step(token, state) → (logits, newState)`

2. **Logits → Frequency Table**
   - Apply softmax to logits to get probabilities
   - Quantize probabilities to integer frequencies (total ≤ 16384)
   - Ensure no frequency is zero (minimum 1)
   - Build cumulative frequency table

3. **NpuBackend Implementation**
   - Load model via MindSpore Lite / HiAI Foundation runtime
   - Maintain hidden state across `update()` calls
   - `reset()` → reinitialize hidden state
   - `getFrequency()` / `findSymbol()` → use current logits-derived frequency table
   - `update(symbol)` → run one model step, update hidden state and frequency table

4. **Archive Compatibility**
   - Set unique backend ID and model ID in archive header
   - Decompressor must have the exact same model to decode
   - Version the model: any weight/quantization change = new model ID

5. **Determinism Requirements**
   - NPU inference must produce bit-identical results across runs
   - If NPU has non-deterministic behavior, use CPU fallback
   - Test with round-trip: compress → decompress → exact match

6. **Fallback**
   - If NPU model not available, fall back to CpuReferenceBackend
   - UI must clearly indicate which backend is active

## HarmonyOS APIs (Expected)

- MindSpore Lite: `@kit.MindSporeLiteKit` or native MindSpore Lite C API via N-API
- HiAI Foundation: device-specific, check `canIUse('SystemCapability.AI.MindSporeLite')`

## Performance Considerations

- Model step latency determines compression speed
- For 64KB chunk with byte tokenizer: 65536 model invocations per chunk
- Target: < 1ms per step for reasonable UX (< 65s per chunk)
- Consider larger token vocabularies (subword) to reduce step count
