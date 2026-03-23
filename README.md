# LexiFold

Neural-inspired text compression for HarmonyOS, inspired by ts_zip and NNCP engineering principles.

## What It Does

LexiFold compresses text using **predictive modeling + arithmetic coding**:
1. A predictor backend estimates probability distributions over the next byte
2. An arithmetic coder encodes each byte using fewer bits when the prediction is confident
3. The same process runs in reverse for lossless decompression

This is the same fundamental approach used by ts_zip (Bellard), but with a modular architecture designed for future NPU/neural model integration on Huawei devices.

## Current Status (v0)

- **Byte-level tokenizer** (256 vocab, UTF-8)
- **CPU Reference Backend** — order-0 adaptive frequency model (real compression, ~60% ratio on English text)
- **24-bit arithmetic coder** — deterministic, bit-level I/O
- **Custom archive format** (`.lxf`) with CRC32 checksums per chunk
- **64KB chunk processing** with independent state reset
- **Round-trip verified** — compress → decompress → exact match confirmed across multiple test cases

### Compression Results (v0 reference backend)

| Input | Size | Compressed | Ratio |
|-------|------|-----------|-------|
| Repeated English text (5.6KB) | 5390 B | 3356 B | 62.3% |
| English paragraph | 123 B | 105 B | 85.4% |
| JSON data | 56 B | 51 B | 91.1% |
| Repeated pattern "abc..." | 30 B | 23 B | 76.7% |

## Architecture

```
UI (Index.ets)
  → CompressionService
    → TextPreprocessor (identity in v0)
    → ByteTokenizer (UTF-8 → bytes)
    → ChunkProcessor (64KB chunks)
      → PredictorBackend.getDistribution()
      → ArithmeticEncoder/Decoder
      → CRC32 verification
    → ArchiveFormat (.lxf read/write)
```

### Key Files

| File | Purpose |
|------|---------|
| `entry/src/main/ets/core/Types.ets` | All types, constants, interfaces |
| `entry/src/main/ets/core/ArithmeticCoder.ets` | Encoder + Decoder |
| `entry/src/main/ets/core/PredictorBackend.ets` | Backend interface + CPU/Mock implementations |
| `entry/src/main/ets/core/ChunkProcessor.ets` | Compress/decompress orchestration |
| `entry/src/main/ets/core/ArchiveFormat.ets` | .lxf container format |
| `entry/src/main/ets/service/CompressionService.ets` | High-level service layer |
| `entry/src/main/ets/pages/Index.ets` | UI |

## Build & Run

Requires **DevEco Studio** with HarmonyOS SDK 6.0.2(22).

1. Open this project in DevEco Studio
2. Build: `Build → Build Hap(s)/APP(s) → Build Hap(s)`
3. Run on device/emulator: `Run → Run 'entry'`

### Verify Core Logic (No Device Needed)

```bash
npx tsx tools/verify_roundtrip.ts
```

## Archive Format (.lxf)

26-byte header + chunk data. See `docs/architecture.md` for full specification.

Header includes: magic (`LXFD`), version, tokenizer/backend/model IDs, chunk size, original size, CRC32.
Each chunk has its own CRC32 for integrity verification.

## Limitations (v0)

- Text-only input (no binary compression)
- Order-0 adaptive model only — no neural prediction yet
- Synchronous compression (may block UI on large files)
- No NPU acceleration

## Roadmap

- **v1**: Reversible text preprocessing, tokenizer profiles, profile switching UI
- **v2**: NPU backend integration via `step(token, state) → logits` interface
- **Experimental**: NNCP-style online adaptive backend (placeholder only)

See `docs/implementation_plan.md` and `docs/npu_integration.md` for details.
