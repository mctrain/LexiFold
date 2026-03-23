// LexiFold Core Types and Constants

// === Archive Constants ===
export const ARCHIVE_MAGIC: string = 'LXFD';
export const ARCHIVE_VERSION: number = 0x01;
export const DEFAULT_CHUNK_SIZE: number = 65536; // 64 KB
export const ARCHIVE_HEADER_SIZE: number = 26;
export const CHUNK_HEADER_SIZE: number = 12;
export const FILE_EXTENSION: string = '.lxf';

// === Arithmetic Coder Constants ===
export const CODE_BITS: number = 24;
export const TOP_VALUE: number = 1 << CODE_BITS;       // 16777216
export const HALF: number = 1 << (CODE_BITS - 1);      // 8388608
export const FIRST_QTR: number = 1 << (CODE_BITS - 2); // 4194304
export const THIRD_QTR: number = HALF + FIRST_QTR;      // 12582912

// === Model Constants ===
export const BYTE_VOCAB_SIZE: number = 256;
export const MAX_TOTAL_FREQ: number = 16384; // rescale threshold

// === Tokenizer IDs ===
export const TOKENIZER_BYTE_256: number = 0x01;

// === Preprocessor IDs ===
export const PREPROCESSOR_IDENTITY: number = 0x00;

// === Backend IDs ===
export const BACKEND_CPU_REFERENCE: number = 0x0001;
export const BACKEND_MOCK_STEP: number = 0x0002;
export const BACKEND_NPU: number = 0x0010;
export const BACKEND_EXPERIMENTAL_ADAPTIVE: number = 0x00FF;

// === Model IDs ===
export const MODEL_NONE: number = 0x0000;
export const MODEL_ORDER0_ADAPTIVE: number = 0x0001;

// === Flags ===
export const FLAG_NONE: number = 0x00;

// === Interfaces ===

export interface FrequencyInfo {
  cumLow: number;
  cumHigh: number;
}

export interface SymbolInfo {
  symbol: number;
  cumLow: number;
  cumHigh: number;
}

export interface PredictorBackend {
  id: number;
  modelId: number;
  name: string;
  vocabSize: number;
  reset(): void;
  getFrequency(symbol: number): FrequencyInfo;
  getTotal(): number;
  findSymbol(scaledValue: number): SymbolInfo;
  update(symbol: number): void;
}

export interface ArchiveHeader {
  magic: string;
  version: number;
  flags: number;
  tokenizerId: number;
  preprocessorId: number;
  backendId: number;
  modelId: number;
  chunkSize: number;
  originalSize: number;
  numChunks: number;
  headerChecksum: number;
}

export interface ChunkMeta {
  compressedSize: number;
  originalSize: number;
  checksum: number;
}

export interface CompressionResult {
  success: boolean;
  originalSize: number;
  compressedSize: number;
  ratio: number;
  timeMs: number;
  outputPath: string;
  backendName: string;
  tokenizerId: number;
  error: string;
}

export interface DecompressionResult {
  success: boolean;
  originalSize: number;
  compressedSize: number;
  timeMs: number;
  text: string;
  backendName: string;
  tokenizerId: number;
  error: string;
}

export interface CompressionProfile {
  tokenizerId: number;
  preprocessorId: number;
  backendId: number;
  modelId: number;
  chunkSize: number;
}

// Compound return types (ArkTS requires named types, no anonymous object types)

export interface CompressWithDataResult {
  result: CompressionResult;
  archiveData: Uint8Array;
}

export interface RoundTripResult {
  success: boolean;
  error: string;
}

export interface ChunkWithData {
  meta: ChunkMeta;
  data: Uint8Array;
}

export function getDefaultProfile(): CompressionProfile {
  return {
    tokenizerId: TOKENIZER_BYTE_256,
    preprocessorId: PREPROCESSOR_IDENTITY,
    backendId: BACKEND_CPU_REFERENCE,
    modelId: MODEL_ORDER0_ADAPTIVE,
    chunkSize: DEFAULT_CHUNK_SIZE
  };
}

export function getBackendName(backendId: number): string {
  if (backendId === BACKEND_CPU_REFERENCE) {
    return 'CPU Reference (Order-0 Adaptive)';
  } else if (backendId === BACKEND_MOCK_STEP) {
    return 'Mock Step Model';
  } else if (backendId === BACKEND_NPU) {
    return 'NPU (Not Available)';
  } else if (backendId === BACKEND_EXPERIMENTAL_ADAPTIVE) {
    return 'Experimental Adaptive (Not Available)';
  }
  return 'Unknown';
}

export function getTokenizerName(tokenizerId: number): string {
  if (tokenizerId === TOKENIZER_BYTE_256) {
    return 'Byte-256';
  }
  return 'Unknown';
}
