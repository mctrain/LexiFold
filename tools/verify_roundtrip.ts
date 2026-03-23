// Standalone round-trip verification for LexiFold core algorithms.
// Run with: npx ts-node tools/verify_roundtrip.ts
// Or: npx tsx tools/verify_roundtrip.ts
//
// This extracts the pure algorithm logic (no HarmonyOS/ArkTS deps) to verify correctness.

// === Constants ===
const CODE_BITS = 24;
const TOP_VALUE = 1 << CODE_BITS;
const HALF = 1 << (CODE_BITS - 1);
const FIRST_QTR = 1 << (CODE_BITS - 2);
const THIRD_QTR = HALF + FIRST_QTR;
const BYTE_VOCAB_SIZE = 256;
const MAX_TOTAL_FREQ = 16384;

// === CRC32 ===
function buildCrc32Table(): number[] {
  const table: number[] = new Array(256);
  for (let i = 0; i < 256; i++) {
    let crc = i;
    for (let j = 0; j < 8; j++) {
      if ((crc & 1) !== 0) {
        crc = ((crc >>> 1) ^ 0xEDB88320) >>> 0;
      } else {
        crc = (crc >>> 1) >>> 0;
      }
    }
    table[i] = crc >>> 0;
  }
  return table;
}
const CRC32_TABLE = buildCrc32Table();

function crc32(data: Uint8Array): number {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < data.length; i++) {
    crc = (CRC32_TABLE[(crc ^ data[i]) & 0xFF] ^ (crc >>> 8)) >>> 0;
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

// === Adaptive Model ===
class AdaptiveModel {
  counts: number[];
  total: number;

  constructor() {
    this.counts = new Array(BYTE_VOCAB_SIZE).fill(1);
    this.total = BYTE_VOCAB_SIZE;
  }

  getFrequency(symbol: number): { cumLow: number; cumHigh: number } {
    let cumLow = 0;
    for (let i = 0; i < symbol; i++) cumLow += this.counts[i];
    return { cumLow, cumHigh: cumLow + this.counts[symbol] };
  }

  getTotal(): number { return this.total; }

  findSymbol(scaledValue: number): { symbol: number; cumLow: number; cumHigh: number } {
    let cumFreq = 0;
    for (let i = 0; i < BYTE_VOCAB_SIZE; i++) {
      const nextCum = cumFreq + this.counts[i];
      if (nextCum > scaledValue) {
        return { symbol: i, cumLow: cumFreq, cumHigh: nextCum };
      }
      cumFreq = nextCum;
    }
    return { symbol: BYTE_VOCAB_SIZE - 1, cumLow: this.total - this.counts[BYTE_VOCAB_SIZE - 1], cumHigh: this.total };
  }

  update(symbol: number): void {
    this.counts[symbol]++;
    this.total++;
    if (this.total >= MAX_TOTAL_FREQ) {
      this.total = 0;
      for (let i = 0; i < BYTE_VOCAB_SIZE; i++) {
        this.counts[i] = Math.max(1, Math.floor(this.counts[i] / 2));
        this.total += this.counts[i];
      }
    }
  }
}

// === Arithmetic Encoder ===
class ArithmeticEncoder {
  private low = 0;
  private high = TOP_VALUE - 1;
  private pendingBits = 0;
  private outputBits: number[] = [];

  encode(cumLow: number, cumHigh: number, total: number): void {
    const range = this.high - this.low + 1;
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
    this.pendingBits++;
    if (this.low < FIRST_QTR) {
      this.emitBitPlusFollow(0);
    } else {
      this.emitBitPlusFollow(1);
    }
    const numBytes = Math.ceil(this.outputBits.length / 8);
    const bytes = new Uint8Array(numBytes);
    for (let i = 0; i < this.outputBits.length; i++) {
      if (this.outputBits[i] === 1) {
        bytes[Math.floor(i / 8)] |= (128 >> (i % 8));
      }
    }
    return bytes;
  }

  private emitBitPlusFollow(bit: number): void {
    this.outputBits.push(bit);
    while (this.pendingBits > 0) {
      this.outputBits.push(1 - bit);
      this.pendingBits--;
    }
  }
}

// === Arithmetic Decoder ===
class ArithmeticDecoder {
  private low = 0;
  private high = TOP_VALUE - 1;
  private value = 0;
  private bitPos = 0;
  private data: Uint8Array;

  constructor(data: Uint8Array) {
    this.data = data;
    for (let i = 0; i < CODE_BITS; i++) {
      this.value = this.value * 2 + this.readBit();
    }
  }

  getScaledValue(total: number): number {
    const range = this.high - this.low + 1;
    return Math.floor(((this.value - this.low + 1) * total - 1) / range);
  }

  decode(cumLow: number, cumHigh: number, total: number): void {
    const range = this.high - this.low + 1;
    this.high = this.low + Math.floor(range * cumHigh / total) - 1;
    this.low = this.low + Math.floor(range * cumLow / total);

    while (true) {
      if (this.high < HALF) {
        // nothing
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
    const byteIdx = Math.floor(this.bitPos / 8);
    const bitIdx = 7 - (this.bitPos % 8);
    this.bitPos++;
    if (byteIdx >= this.data.length) return 0;
    return (this.data[byteIdx] >> bitIdx) & 1;
  }
}

// === Compress ===
function compress(input: Uint8Array): Uint8Array {
  const model = new AdaptiveModel();
  const encoder = new ArithmeticEncoder();

  for (let i = 0; i < input.length; i++) {
    const symbol = input[i];
    const total = model.getTotal();
    const freq = model.getFrequency(symbol);
    encoder.encode(freq.cumLow, freq.cumHigh, total);
    model.update(symbol);
  }

  return encoder.finish();
}

// === Decompress ===
function decompress(compressed: Uint8Array, originalSize: number): Uint8Array {
  const model = new AdaptiveModel();
  const decoder = new ArithmeticDecoder(compressed);
  const output = new Uint8Array(originalSize);

  for (let i = 0; i < originalSize; i++) {
    const total = model.getTotal();
    const scaledValue = decoder.getScaledValue(total);
    const info = model.findSymbol(scaledValue);
    decoder.decode(info.cumLow, info.cumHigh, total);
    model.update(info.symbol);
    output[i] = info.symbol;
  }

  return output;
}

// === Tests ===
function arraysEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function testRoundTrip(name: string, input: Uint8Array): void {
  const compressed = compress(input);
  const decompressed = decompress(compressed, input.length);
  const match = arraysEqual(input, decompressed);
  const ratio = compressed.length / input.length;
  const crcOrig = crc32(input);
  const crcDecomp = crc32(decompressed);

  console.log(`[${match ? 'PASS' : 'FAIL'}] ${name}`);
  console.log(`  Original: ${input.length} bytes, CRC32: 0x${crcOrig.toString(16).padStart(8, '0')}`);
  console.log(`  Compressed: ${compressed.length} bytes (${(ratio * 100).toFixed(1)}%)`);
  console.log(`  Decompressed CRC32: 0x${crcDecomp.toString(16).padStart(8, '0')}`);
  if (!match) {
    console.log('  MISMATCH DETECTED!');
    // Find first difference
    for (let i = 0; i < Math.max(input.length, decompressed.length); i++) {
      if (i >= input.length || i >= decompressed.length || input[i] !== decompressed[i]) {
        console.log(`  First diff at byte ${i}: expected ${input[i]}, got ${decompressed[i]}`);
        break;
      }
    }
    process.exit(1);
  }
}

// Run tests
console.log('=== LexiFold Round-Trip Verification ===\n');

// Test 1: Empty-ish (single byte)
testRoundTrip('Single byte', new Uint8Array([65]));

// Test 2: Short ASCII
const encoder = new TextEncoder();
testRoundTrip('Short ASCII', encoder.encode('Hello, World!'));

// Test 3: Repeated text (should compress well)
testRoundTrip('Repeated text', encoder.encode('abcabcabcabcabcabcabcabcabcabc'));

// Test 4: English paragraph
const english = 'The quick brown fox jumps over the lazy dog. ' +
  'Pack my box with five dozen liquor jugs. ' +
  'How vexingly quick daft zebras jump! ';
testRoundTrip('English paragraph', encoder.encode(english));

// Test 5: All byte values
const allBytes = new Uint8Array(256);
for (let i = 0; i < 256; i++) allBytes[i] = i;
testRoundTrip('All byte values', allBytes);

// Test 6: Larger text (multiple repetitions)
let largeText = '';
for (let i = 0; i < 100; i++) {
  largeText += `Line ${i}: The quick brown fox jumps over the lazy dog.\n`;
}
testRoundTrip('Large repeated text (5.6KB)', encoder.encode(largeText));

// Test 7: Random-ish data (should not compress much)
const random = new Uint8Array(1000);
let seed = 42;
for (let i = 0; i < 1000; i++) {
  seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
  random[i] = (seed >> 16) & 0xFF;
}
testRoundTrip('Pseudo-random data (1KB)', random);

// Test 8: UTF-8 multibyte
testRoundTrip('UTF-8 Chinese', encoder.encode('你好世界！这是一段中文文本。'));

// Test 9: JSON-like
testRoundTrip('JSON data', encoder.encode('{"name":"LexiFold","version":"1.0","type":"compression"}'));

console.log('\n=== All tests passed! ===');
