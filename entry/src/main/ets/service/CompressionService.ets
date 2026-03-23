// Compression Service — orchestrates compress/decompress tasks for the UI layer

import {
  CompressionProfile, CompressionResult, DecompressionResult,
  CompressWithDataResult, RoundTripResult, Preprocessor,
  BackendOption, PreprocessorOption,
  getDefaultProfile, getBackendName, getTokenizerName, getPreprocessorName,
  getAvailableBackends, getAvailablePreprocessors,
  FILE_EXTENSION
} from '../core/Types';
import { ByteTokenizer } from '../core/Tokenizer';
import { createPreprocessor } from '../core/TextPreprocessor';
import { compressData, decompressData, CompressOutput, DecompressOutput } from '../core/ChunkProcessor';

function makeCompressionResult(
  success: boolean, originalSize: number, compressedSize: number,
  ratio: number, timeMs: number, backendName: string,
  tokenizerId: number, error: string
): CompressionResult {
  const r: CompressionResult = {
    success: success,
    originalSize: originalSize,
    compressedSize: compressedSize,
    ratio: ratio,
    timeMs: timeMs,
    outputPath: '',
    backendName: backendName,
    tokenizerId: tokenizerId,
    error: error
  };
  return r;
}

function makeDecompressionResult(
  success: boolean, originalSize: number, compressedSize: number,
  timeMs: number, text: string, backendName: string,
  tokenizerId: number, error: string
): DecompressionResult {
  const r: DecompressionResult = {
    success: success,
    originalSize: originalSize,
    compressedSize: compressedSize,
    timeMs: timeMs,
    text: text,
    backendName: backendName,
    tokenizerId: tokenizerId,
    error: error
  };
  return r;
}

export class CompressionService {
  private profile: CompressionProfile;
  private tokenizer: ByteTokenizer;
  private preprocessor: Preprocessor;

  constructor() {
    this.profile = getDefaultProfile();
    this.tokenizer = new ByteTokenizer();
    this.preprocessor = createPreprocessor(this.profile.preprocessorId);
  }

  // === Profile management ===

  getProfile(): CompressionProfile {
    return this.profile;
  }

  setBackend(backendId: number, modelId: number): void {
    this.profile = {
      tokenizerId: this.profile.tokenizerId,
      preprocessorId: this.profile.preprocessorId,
      backendId: backendId,
      modelId: modelId,
      chunkSize: this.profile.chunkSize
    };
  }

  setPreprocessor(preprocessorId: number): void {
    this.profile = {
      tokenizerId: this.profile.tokenizerId,
      preprocessorId: preprocessorId,
      backendId: this.profile.backendId,
      modelId: this.profile.modelId,
      chunkSize: this.profile.chunkSize
    };
    this.preprocessor = createPreprocessor(preprocessorId);
  }

  getBackendDisplayName(): string {
    return getBackendName(this.profile.backendId);
  }

  getTokenizerDisplayName(): string {
    return getTokenizerName(this.profile.tokenizerId);
  }

  getPreprocessorDisplayName(): string {
    return getPreprocessorName(this.profile.preprocessorId);
  }

  getFileExtension(): string {
    return FILE_EXTENSION;
  }

  getBackendOptions(): BackendOption[] {
    return getAvailableBackends();
  }

  getPreprocessorOptions(): PreprocessorOption[] {
    return getAvailablePreprocessors();
  }

  // === Compress text → CompressionResult only ===

  compressText(text: string): CompressionResult {
    const startTime: number = Date.now();
    try {
      const processed: string = this.preprocessor.preprocess(text);
      const inputBytes: Uint8Array = this.tokenizer.tokenize(processed);
      if (inputBytes.length === 0) {
        return makeCompressionResult(false, 0, 0, 0, Date.now() - startTime,
          this.getBackendDisplayName(), this.profile.tokenizerId, 'Input text is empty');
      }
      const output: CompressOutput = compressData(inputBytes, this.profile);
      const elapsed: number = Date.now() - startTime;
      return makeCompressionResult(true, output.originalSize, output.compressedSize,
        output.compressedSize / output.originalSize, elapsed,
        this.getBackendDisplayName(), this.profile.tokenizerId, '');
    } catch (e) {
      return makeCompressionResult(false, 0, 0, 0, Date.now() - startTime,
        this.getBackendDisplayName(), this.profile.tokenizerId,
        `Compression failed: ${(e as Error).message || String(e)}`);
    }
  }

  // === Compress text → archive bytes ===

  compressTextToBytes(text: string): CompressWithDataResult {
    const startTime: number = Date.now();
    try {
      const processed: string = this.preprocessor.preprocess(text);
      const inputBytes: Uint8Array = this.tokenizer.tokenize(processed);
      if (inputBytes.length === 0) {
        const r: CompressWithDataResult = {
          result: makeCompressionResult(false, 0, 0, 0, Date.now() - startTime,
            this.getBackendDisplayName(), this.profile.tokenizerId, 'Input text is empty'),
          archiveData: new Uint8Array(0)
        };
        return r;
      }
      const output: CompressOutput = compressData(inputBytes, this.profile);
      const elapsed: number = Date.now() - startTime;
      const r: CompressWithDataResult = {
        result: makeCompressionResult(true, output.originalSize, output.compressedSize,
          output.compressedSize / output.originalSize, elapsed,
          this.getBackendDisplayName(), this.profile.tokenizerId, ''),
        archiveData: output.archiveData
      };
      return r;
    } catch (e) {
      const r: CompressWithDataResult = {
        result: makeCompressionResult(false, 0, 0, 0, Date.now() - startTime,
          this.getBackendDisplayName(), this.profile.tokenizerId,
          `Compression failed: ${(e as Error).message || String(e)}`),
        archiveData: new Uint8Array(0)
      };
      return r;
    }
  }

  // === Compress raw bytes → archive bytes ===

  compressBytes(inputBytes: Uint8Array): CompressWithDataResult {
    const startTime: number = Date.now();
    try {
      if (inputBytes.length === 0) {
        const r: CompressWithDataResult = {
          result: makeCompressionResult(false, 0, 0, 0, Date.now() - startTime,
            this.getBackendDisplayName(), this.profile.tokenizerId, 'Input is empty'),
          archiveData: new Uint8Array(0)
        };
        return r;
      }
      const output: CompressOutput = compressData(inputBytes, this.profile);
      const elapsed: number = Date.now() - startTime;
      const r: CompressWithDataResult = {
        result: makeCompressionResult(true, output.originalSize, output.compressedSize,
          output.compressedSize / output.originalSize, elapsed,
          this.getBackendDisplayName(), this.profile.tokenizerId, ''),
        archiveData: output.archiveData
      };
      return r;
    } catch (e) {
      const r: CompressWithDataResult = {
        result: makeCompressionResult(false, 0, 0, 0, Date.now() - startTime,
          this.getBackendDisplayName(), this.profile.tokenizerId,
          `Compression failed: ${(e as Error).message || String(e)}`),
        archiveData: new Uint8Array(0)
      };
      return r;
    }
  }

  // === Decompress archive bytes → text ===

  decompressToText(archiveData: Uint8Array): DecompressionResult {
    const startTime: number = Date.now();
    try {
      const output: DecompressOutput = decompressData(archiveData);
      if (output.error !== '') {
        return makeDecompressionResult(false, output.originalSize, output.compressedSize,
          Date.now() - startTime, '', this.getBackendDisplayName(),
          this.profile.tokenizerId, output.error);
      }
      const rawText: string = this.tokenizer.detokenize(output.data);
      // Use the preprocessor recorded in the archive for correct reversal
      const text: string = this.preprocessor.reverse(rawText);
      return makeDecompressionResult(true, output.originalSize, output.compressedSize,
        Date.now() - startTime, text, this.getBackendDisplayName(),
        this.profile.tokenizerId, '');
    } catch (e) {
      return makeDecompressionResult(false, 0, archiveData.length,
        Date.now() - startTime, '', this.getBackendDisplayName(),
        this.profile.tokenizerId,
        `Decompression failed: ${(e as Error).message || String(e)}`);
    }
  }

  // === Round-trip verification ===

  verifyRoundTrip(text: string): RoundTripResult {
    const compressed: CompressWithDataResult = this.compressTextToBytes(text);
    if (!compressed.result.success) {
      const r: RoundTripResult = { success: false, error: `Compress failed: ${compressed.result.error}` };
      return r;
    }
    const decompressed: DecompressionResult = this.decompressToText(compressed.archiveData);
    if (!decompressed.success) {
      const r: RoundTripResult = { success: false, error: `Decompress failed: ${decompressed.error}` };
      return r;
    }
    if (decompressed.text !== text) {
      const r: RoundTripResult = { success: false, error: 'Round-trip mismatch: decompressed text differs from original' };
      return r;
    }
    const r: RoundTripResult = { success: true, error: '' };
    return r;
  }
}
