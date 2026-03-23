// Arithmetic Encoder and Decoder for LexiFold
// Uses 24-bit precision, bit-level I/O, classic Witten-Neal-Cleary style.
// Deterministic: identical encode/decode paths guarantee lossless round-trip.

import {
  CODE_BITS, TOP_VALUE, HALF, FIRST_QTR, THIRD_QTR
} from './Types';

// === Arithmetic Encoder ===

export class ArithmeticEncoder {
  private low: number = 0;
  private high: number = TOP_VALUE - 1;
  private pendingBits: number = 0;
  private outputBits: number[] = [];

  encode(cumLow: number, cumHigh: number, total: number): void {
    const range: number = this.high - this.low + 1;
    this.high = this.low + Math.floor(range * cumHigh / total) - 1;
    this.low = this.low + Math.floor(range * cumLow / total);

    while (true) {
      if (this.high < HALF) {
        this.emitBitPlusFollow(0);
      } else if (this.low >= HALF) {
        this.emitBitPlusFollow(1);
        this.low -= HALF;
        this.high -= HALF;
      } else if (this.low >= FIRST_QTR && this.high < THIRD_QTR) {
        this.pendingBits++;
        this.low -= FIRST_QTR;
        this.high -= FIRST_QTR;
      } else {
        break;
      }
      this.low = this.low * 2;
      this.high = this.high * 2 + 1;
    }
  }

  finish(): Uint8Array {
    // Output bits to disambiguate the final interval
    this.pendingBits++;
    if (this.low < FIRST_QTR) {
      this.emitBitPlusFollow(0);
    } else {
      this.emitBitPlusFollow(1);
    }
    return this.packBitsToBytes();
  }

  reset(): void {
    this.low = 0;
    this.high = TOP_VALUE - 1;
    this.pendingBits = 0;
    this.outputBits = [];
  }

  private emitBitPlusFollow(bit: number): void {
    this.outputBits.push(bit);
    while (this.pendingBits > 0) {
      this.outputBits.push(1 - bit);
      this.pendingBits--;
    }
  }

  private packBitsToBytes(): Uint8Array {
    const numBytes: number = Math.ceil(this.outputBits.length / 8);
    const bytes = new Uint8Array(numBytes);
    for (let i = 0; i < this.outputBits.length; i++) {
      if (this.outputBits[i] === 1) {
        const byteIdx: number = Math.floor(i / 8);
        const bitIdx: number = 7 - (i % 8);
        bytes[byteIdx] = bytes[byteIdx] | (1 << bitIdx);
      }
    }
    return bytes;
  }
}

// === Arithmetic Decoder ===

export class ArithmeticDecoder {
  private low: number = 0;
  private high: number = TOP_VALUE - 1;
  private value: number = 0;
  private bitPos: number = 0;
  private data: Uint8Array;

  constructor(data: Uint8Array) {
    this.data = data;
    // Initialize value register with first CODE_BITS bits
    for (let i = 0; i < CODE_BITS; i++) {
      this.value = this.value * 2 + this.readBit();
    }
  }

  // Get the scaled frequency value to look up the symbol
  getScaledValue(total: number): number {
    const range: number = this.high - this.low + 1;
    return Math.floor(((this.value - this.low + 1) * total - 1) / range);
  }

  // After finding the symbol and its cumulative frequencies, update decoder state
  decode(cumLow: number, cumHigh: number, total: number): void {
    const range: number = this.high - this.low + 1;
    this.high = this.low + Math.floor(range * cumHigh / total) - 1;
    this.low = this.low + Math.floor(range * cumLow / total);

    while (true) {
      if (this.high < HALF) {
        // Both in lower half — nothing to adjust
      } else if (this.low >= HALF) {
        this.value -= HALF;
        this.low -= HALF;
        this.high -= HALF;
      } else if (this.low >= FIRST_QTR && this.high < THIRD_QTR) {
        this.value -= FIRST_QTR;
        this.low -= FIRST_QTR;
        this.high -= FIRST_QTR;
      } else {
        break;
      }
      this.low = this.low * 2;
      this.high = this.high * 2 + 1;
      this.value = this.value * 2 + this.readBit();
    }
  }

  private readBit(): number {
    const byteIdx: number = Math.floor(this.bitPos / 8);
    const bitIdx: number = 7 - (this.bitPos % 8);
    this.bitPos++;
    if (byteIdx >= this.data.length) {
      return 0;
    }
    return (this.data[byteIdx] >> bitIdx) & 1;
  }
}
