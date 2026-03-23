# LexiFold Implementation Plan

## Overview

LexiFold is a text compression POC for HarmonyOS, inspired by ts_zip (fixed model inference + arithmetic coding)
and borrowing engineering ideas from NNCP (profiling, preprocessing, encode/decode symmetry).

## Architecture Layers

1. **UI Layer** — `entry/src/main/ets/pages/Index.ets`
2. **Service Layer** — `entry/src/main/ets/service/CompressionService.ets`
3. **Core Layer** — `entry/src/main/ets/core/*.ets`
4. **Backend Layer** — embedded in `PredictorBackend.ets` (interface + implementations)

## File Plan

### New Files (v0)

| File | Purpose |
|------|---------|
| `core/Types.ets` | Shared types, enums, constants, config interfaces |
| `core/Checksum.ets` | CRC32 implementation |
| `core/ArithmeticCoder.ets` | Arithmetic encoder + decoder (bit-level, 24-bit precision) |
| `core/PredictorBackend.ets` | Backend interface + CpuReferenceBackend (order-0 adaptive) + MockStepModelBackend |
| `core/Tokenizer.ets` | ByteTokenizer (256 vocab, identity mapping) |
| `core/TextPreprocessor.ets` | Identity preprocessor (v0), reversible text preprocessor (v1) |
| `core/ArchiveFormat.ets` | .lxf archive header read/write, magic/version/metadata |
| `core/ChunkProcessor.ets` | Chunk-based compress/decompress orchestration |
| `service/CompressionService.ets` | High-level task orchestration, file I/O, error mapping |
| `docs/architecture.md` | Architecture documentation |
| `docs/npu_integration.md` | NPU integration guide (placeholder) |

### Modified Files

| File | Change |
|------|--------|
| `pages/Index.ets` | Full rewrite: compress/decompress UI with metrics display |
| `resources/base/element/string.json` | Add app strings |

## Version Boundaries

### v0 (Current target)
- Byte-level tokenizer (256 vocab)
- CpuReferenceBackend (order-0 adaptive frequency model)
- 24-bit arithmetic coder
- .lxf archive format with CRC32 checksums
- 64KB chunks
- Compress text input → .lxf file
- Select text file → compress → .lxf file
- Select .lxf file → decompress → verify exact match
- UI: backend/tokenizer info, size/ratio/time metrics

### v1 (Future)
- TextPreprocessor with reversible case normalization
- TokenizerProfile system (byte-256, text-small-vocab, experimental-subword)
- Profile switching UI
- Stricter validation and error handling

### v2 (Future)
- NPU backend integration via step(token, state) → logits interface
- Real small model (RWKV-style) deployment
- Performance optimization (native arithmetic coder if needed)

### Experimental (Placeholder only)
- ExperimentalAdaptiveBackend — for NNCP-style online adaptation (no training in v0/v1)

## Key Design Decisions

1. **24-bit arithmetic coder** — avoids JS/ArkTS 32-bit signed integer issues, stays within safe integer range
2. **Bit-level output** — simpler carry handling than byte-output range coder, optimize in v1
3. **Order-0 adaptive model** — real compression (~60% for English), deterministic, symmetric encode/decode
4. **CRC32 per chunk** — catches any encode/decode mismatch immediately
5. **No native layer in v0** — ArkTS sufficient for correctness; profile before optimizing
