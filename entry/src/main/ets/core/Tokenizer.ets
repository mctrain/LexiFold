// Tokenizer for LexiFold
// v0: byte-level tokenizer (256 vocab, identity mapping)
// Future: subword tokenizers with reversible encoding

import { util } from '@kit.ArkTS';
import { TOKENIZER_BYTE_256 } from './Types';

export class ByteTokenizer {
  readonly id: number = TOKENIZER_BYTE_256;
  readonly name: string = 'Byte-256';
  readonly vocabSize: number = 256;

  // Tokenize: convert string to byte array (UTF-8)
  tokenize(text: string): Uint8Array {
    const encoder = new util.TextEncoder();
    return encoder.encodeInto(text);
  }

  // Detokenize: convert byte array back to string (UTF-8)
  detokenize(data: Uint8Array): string {
    const decoder = util.TextDecoder.create('utf-8');
    return decoder.decodeToString(data);
  }

  // Token is the byte value itself (identity mapping)
  byteToToken(byte: number): number {
    return byte & 0xFF;
  }

  tokenToByte(token: number): number {
    return token & 0xFF;
  }
}
