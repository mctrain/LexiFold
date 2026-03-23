# LexiFold Architecture

## Compression Pipeline

```
Input Text
  → TextPreprocessor (identity in v0)
  → Tokenizer (byte-level, 256 vocab)
  → ChunkProcessor (64KB chunks)
    → For each chunk:
      → PredictorBackend.reset()
      → For each token:
        → backend.getDistribution() → probability table
        → ArithmeticEncoder.encode(token, probabilities)
        → backend.update(token)
      → encoder.finish() → compressed bits
      → CRC32(original chunk) → checksum
  → ArchiveFormat.write(header + chunks) → .lxf file
```

## Decompression Pipeline

```
.lxf file
  → ArchiveFormat.read() → header + chunks
  → Validate header (magic, version, backend/tokenizer compatibility)
  → ChunkProcessor (per chunk):
    → PredictorBackend.reset()
    → For each position (0..originalChunkSize-1):
      → backend.getDistribution() → probability table
      → ArithmeticDecoder.decode(probabilities) → token
      → backend.update(token)
    → Verify CRC32
  → Tokenizer.detokenize() (identity for bytes)
  → TextPreprocessor.reverse() (identity in v0)
  → Output Text
```

## Archive Format (.lxf)

### Header (26 bytes)
| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic: "LXFD" |
| 4 | 1 | Version (0x01) |
| 5 | 1 | Flags |
| 6 | 1 | Tokenizer ID |
| 7 | 1 | Preprocessor ID |
| 8 | 2 | Backend ID (LE) |
| 10 | 2 | Model ID (LE) |
| 12 | 4 | Chunk Size (LE) |
| 16 | 4 | Original Total Size (LE) |
| 20 | 2 | Num Chunks (LE) |
| 22 | 4 | Header CRC32 (of bytes 0-21) |

### Per Chunk
| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Compressed Size (LE) |
| 4 | 4 | Original Chunk Size (LE) |
| 8 | 4 | Original Data CRC32 |
| 12 | N | Compressed Data |

## Determinism Contract

The compress/decompress round-trip is lossless if and only if:
1. Same tokenizer produces same token sequence from same bytes
2. Same backend produces same probability distribution at each step
3. Arithmetic coder uses identical precision and normalization on both sides
4. Backend state evolution is identical (same update() calls in same order)

This is guaranteed by design: encoder and decoder share the same code paths for
prediction and state update. The only asymmetry is in the coder itself (encode vs decode),
which is mathematically symmetric by construction.

## Backend Interface

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

### Implementations
- **CpuReferenceBackend** — Order-0 adaptive frequency model. Real compression.
- **MockStepModelBackend** — Placeholder for step-model interface testing.
- **NpuBackend** — Future: wraps HiAI/MindSpore Lite step inference.
- **ExperimentalAdaptiveBackend** — Future: NNCP-style online adaptation.
