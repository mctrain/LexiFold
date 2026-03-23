// Chunk-based compress/decompress for LexiFold
// Splits input into chunks, compresses each independently, assembles archive.

import {
  ArchiveHeader, ChunkMeta, ChunkWithData, CompressionProfile, PredictorBackend,
  ARCHIVE_MAGIC, ARCHIVE_VERSION, FLAG_NONE,
  DEFAULT_CHUNK_SIZE
} from './Types';
import { crc32 } from './Checksum';
import { ArithmeticEncoder, ArithmeticDecoder } from './ArithmeticCoder';
import { createBackend } from './PredictorBackend';
import { assembleArchive, parseArchive, ParsedArchive } from './ArchiveFormat';

// === Compress a single chunk ===

function compressChunk(data: Uint8Array, backend: PredictorBackend): Uint8Array {
  backend.reset();
  const encoder = new ArithmeticEncoder();

  for (let i = 0; i < data.length; i++) {
    const symbol: number = data[i];
    const total: number = backend.getTotal();
    const freq = backend.getFrequency(symbol);
    encoder.encode(freq.cumLow, freq.cumHigh, total);
    backend.update(symbol);
  }

  return encoder.finish();
}

// === Decompress a single chunk ===

function decompressChunk(
  compressedData: Uint8Array,
  originalSize: number,
  backend: PredictorBackend
): Uint8Array {
  backend.reset();
  const decoder = new ArithmeticDecoder(compressedData);
  const output = new Uint8Array(originalSize);

  for (let i = 0; i < originalSize; i++) {
    const total: number = backend.getTotal();
    const scaledValue: number = decoder.getScaledValue(total);
    const info = backend.findSymbol(scaledValue);
    decoder.decode(info.cumLow, info.cumHigh, total);
    backend.update(info.symbol);
    output[i] = info.symbol;
  }

  return output;
}

// === Full compression pipeline ===

export interface CompressOutput {
  archiveData: Uint8Array;
  originalSize: number;
  compressedSize: number;
}

export function compressData(
  inputData: Uint8Array,
  profile: CompressionProfile
): CompressOutput {
  const chunkSize: number = profile.chunkSize > 0 ? profile.chunkSize : DEFAULT_CHUNK_SIZE;
  const numChunks: number = Math.max(1, Math.ceil(inputData.length / chunkSize));
  const backend: PredictorBackend = createBackend(profile.backendId);

  const chunks: ChunkWithData[] = [];

  for (let i = 0; i < numChunks; i++) {
    const start: number = i * chunkSize;
    const end: number = Math.min(start + chunkSize, inputData.length);
    const chunkData: Uint8Array = inputData.slice(start, end);

    const originalChecksum: number = crc32(chunkData);
    const compressedData: Uint8Array = compressChunk(chunkData, backend);

    const meta: ChunkMeta = {
      compressedSize: compressedData.length,
      originalSize: chunkData.length,
      checksum: originalChecksum
    };

    const chunkEntry: ChunkWithData = { meta: meta, data: compressedData };
    chunks.push(chunkEntry);
  }

  const header: ArchiveHeader = {
    magic: ARCHIVE_MAGIC,
    version: ARCHIVE_VERSION,
    flags: FLAG_NONE,
    tokenizerId: profile.tokenizerId,
    preprocessorId: profile.preprocessorId,
    backendId: profile.backendId,
    modelId: profile.modelId,
    chunkSize: chunkSize,
    originalSize: inputData.length,
    numChunks: numChunks,
    headerChecksum: 0 // Will be computed during assembly
  };

  const archiveData: Uint8Array = assembleArchive(header, chunks);

  return {
    archiveData,
    originalSize: inputData.length,
    compressedSize: archiveData.length
  };
}

// === Full decompression pipeline ===

export interface DecompressOutput {
  data: Uint8Array;
  originalSize: number;
  compressedSize: number;
  error: string;
}

export function decompressData(archiveData: Uint8Array): DecompressOutput {
  const parsed: ParsedArchive | null = parseArchive(archiveData);
  if (parsed === null) {
    return {
      data: new Uint8Array(0),
      originalSize: 0,
      compressedSize: archiveData.length,
      error: 'Invalid archive: header parse failed (bad magic, version, or checksum)'
    };
  }

  const header = parsed.header;
  const backend: PredictorBackend = createBackend(header.backendId);

  // Verify chunk count
  if (parsed.chunks.length !== header.numChunks) {
    return {
      data: new Uint8Array(0),
      originalSize: header.originalSize,
      compressedSize: archiveData.length,
      error: `Chunk count mismatch: expected ${header.numChunks}, got ${parsed.chunks.length}`
    };
  }

  // Decompress all chunks
  const outputParts: Uint8Array[] = [];
  let totalOriginal: number = 0;

  for (let i = 0; i < parsed.chunks.length; i++) {
    const chunk = parsed.chunks[i];

    const decompressed: Uint8Array = decompressChunk(
      chunk.data,
      chunk.meta.originalSize,
      backend
    );

    // Verify CRC32
    const computedChecksum: number = crc32(decompressed);
    if (computedChecksum !== chunk.meta.checksum) {
      return {
        data: new Uint8Array(0),
        originalSize: header.originalSize,
        compressedSize: archiveData.length,
        error: `Chunk ${i} CRC32 mismatch: expected 0x${chunk.meta.checksum.toString(16)}, got 0x${computedChecksum.toString(16)}`
      };
    }

    outputParts.push(decompressed);
    totalOriginal += decompressed.length;
  }

  // Verify total size
  if (totalOriginal !== header.originalSize) {
    return {
      data: new Uint8Array(0),
      originalSize: header.originalSize,
      compressedSize: archiveData.length,
      error: `Size mismatch: expected ${header.originalSize}, got ${totalOriginal}`
    };
  }

  // Concatenate chunks
  const result = new Uint8Array(totalOriginal);
  let offset: number = 0;
  for (const part of outputParts) {
    result.set(part, offset);
    offset += part.length;
  }

  return {
    data: result,
    originalSize: totalOriginal,
    compressedSize: archiveData.length,
    error: ''
  };
}
