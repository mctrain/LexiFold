// Benchmark: compare compression across backends and reference compressors
// Run: npx tsx tools/benchmark.ts

import { execSync } from 'child_process';
import { writeFileSync, unlinkSync, statSync } from 'fs';

// === Constants (mirror from core) ===
const CODE_BITS = 24;
const TOP_VALUE = 1 << CODE_BITS;
const HALF = 1 << (CODE_BITS - 1);
const FIRST_QTR = 1 << (CODE_BITS - 2);
const THIRD_QTR = HALF + FIRST_QTR;
const BYTE_VOCAB_SIZE = 256;
const MAX_TOTAL_FREQ = 16384;

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
      if (this.high < HALF) { this.emit(0); }
      else if (this.low >= HALF) { this.emit(1); this.low -= HALF; this.high -= HALF; }
      else if (this.low >= FIRST_QTR && this.high < THIRD_QTR) { this.pendingBits++; this.low -= FIRST_QTR; this.high -= FIRST_QTR; }
      else break;
      this.low *= 2; this.high = this.high * 2 + 1;
    }
  }
  finish(): number {
    this.pendingBits++;
    if (this.low < FIRST_QTR) this.emit(0); else this.emit(1);
    return Math.ceil(this.outputBits.length / 8);
  }
  private emit(bit: number): void {
    this.outputBits.push(bit);
    while (this.pendingBits > 0) { this.outputBits.push(1 - bit); this.pendingBits--; }
  }
}

// === Backends ===
interface Backend {
  name: string;
  reset(): void;
  getFrequency(sym: number): { cumLow: number; cumHigh: number };
  getTotal(): number;
  update(sym: number): void;
}

class Order0Adaptive implements Backend {
  name = 'Order-0 Adaptive';
  private counts = new Array(256).fill(1);
  private total = 256;
  reset() { this.counts = new Array(256).fill(1); this.total = 256; }
  getFrequency(sym: number) {
    let cumLow = 0;
    for (let i = 0; i < sym; i++) cumLow += this.counts[i];
    return { cumLow, cumHigh: cumLow + this.counts[sym] };
  }
  getTotal() { return this.total; }
  update(sym: number) {
    this.counts[sym]++; this.total++;
    if (this.total >= MAX_TOTAL_FREQ) {
      this.total = 0;
      for (let i = 0; i < 256; i++) { this.counts[i] = Math.max(1, Math.floor(this.counts[i] / 2)); this.total += this.counts[i]; }
    }
  }
}

class Order1Adaptive implements Backend {
  name = 'Order-1 Adaptive';
  private contexts: number[][] = [];
  private totals: number[] = [];
  private prev = 0;
  reset() {
    this.contexts = [];
    this.totals = [];
    for (let i = 0; i < 256; i++) {
      this.contexts.push(new Array(256).fill(1));
      this.totals.push(256);
    }
    this.prev = 0;
  }
  getFrequency(sym: number) {
    const ctx = this.contexts[this.prev];
    let cumLow = 0;
    for (let i = 0; i < sym; i++) cumLow += ctx[i];
    return { cumLow, cumHigh: cumLow + ctx[sym] };
  }
  getTotal() { return this.totals[this.prev]; }
  update(sym: number) {
    this.contexts[this.prev][sym]++;
    this.totals[this.prev]++;
    if (this.totals[this.prev] >= MAX_TOTAL_FREQ) {
      this.totals[this.prev] = 0;
      for (let i = 0; i < 256; i++) {
        this.contexts[this.prev][i] = Math.max(1, Math.floor(this.contexts[this.prev][i] / 2));
        this.totals[this.prev] += this.contexts[this.prev][i];
      }
    }
    this.prev = sym;
  }
}

class UniformBackend implements Backend {
  name = 'Uniform (no model)';
  reset() {}
  getFrequency(sym: number) { return { cumLow: sym, cumHigh: sym + 1 }; }
  getTotal() { return 256; }
  update(_sym: number) {}
}

// === Compress with backend ===
function compressSize(data: Uint8Array, backend: Backend): number {
  backend.reset();
  const encoder = new ArithmeticEncoder();
  for (let i = 0; i < data.length; i++) {
    const sym = data[i];
    const total = backend.getTotal();
    const freq = backend.getFrequency(sym);
    encoder.encode(freq.cumLow, freq.cumHigh, total);
    backend.update(sym);
  }
  return encoder.finish();
}

// === gzip reference ===
function gzipSize(data: Uint8Array): number {
  const tmp = '/tmp/_lexifold_bench.tmp';
  writeFileSync(tmp, data);
  execSync(`gzip -9 -k -f ${tmp}`);
  const size = statSync(tmp + '.gz').size;
  try { unlinkSync(tmp); } catch {}
  try { unlinkSync(tmp + '.gz'); } catch {}
  return size;
}

// === Test data ===
const enc = new TextEncoder();

let english = '';
for (let i = 0; i < 100; i++) english += `Line ${i}: The quick brown fox jumps over the lazy dog.\n`;
const englishData = enc.encode(english);

const shortText = enc.encode('The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump!');

let jsonStr = '';
for (let i = 0; i < 50; i++) jsonStr += `{"id":${i},"name":"item_${i}","value":${Math.floor(Math.random()*1000)},"active":${i%2===0}}\n`;
const jsonData = enc.encode(jsonStr);

const chineseText = '今天天气真好，适合出去散步。春风拂面，花香四溢，令人心旷神怡。这是一段用于测试中文文本压缩效果的示例内容，包含常见的中文字符和标点符号。';
let chineseLong = '';
for (let i = 0; i < 20; i++) chineseLong += chineseText;
const chineseData = enc.encode(chineseLong);

let sourceCode = '';
for (let i = 0; i < 30; i++) {
  sourceCode += `function calc_${i}(a: number, b: number): number {\n`;
  sourceCode += `  const result = a * ${i} + b;\n`;
  sourceCode += `  if (result > 0) { return result; }\n`;
  sourceCode += `  return -result;\n}\n\n`;
}
const sourceData = enc.encode(sourceCode);

const random = new Uint8Array(4096);
let seed = 42;
for (let i = 0; i < 4096; i++) { seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF; random[i] = (seed >> 16) & 0xFF; }

interface TestCase {
  name: string;
  data: Uint8Array;
}

const testCases: TestCase[] = [
  { name: 'English text (5.4KB)', data: englishData },
  { name: 'Short English (123B)', data: shortText },
  { name: 'JSON lines (3.2KB)', data: jsonData },
  { name: 'Chinese text (8.7KB)', data: chineseData },
  { name: 'Source code (3.8KB)', data: sourceData },
  { name: 'Random bytes (4KB)', data: random },
];

const backends: Backend[] = [
  new UniformBackend(),
  new Order0Adaptive(),
  new Order1Adaptive(),
];

// === Run benchmark ===
console.log('# LexiFold Compression Benchmark\n');

// Header
const hdr = ['Test Data', 'Size', ...backends.map(b => b.name), 'gzip -9'];
console.log('| ' + hdr.join(' | ') + ' |');
console.log('| ' + hdr.map(() => '---').join(' | ') + ' |');

for (const tc of testCases) {
  const cols: string[] = [tc.name, `${tc.data.length}`];
  for (const b of backends) {
    const sz = compressSize(tc.data, b);
    const ratio = (sz / tc.data.length * 100).toFixed(1);
    cols.push(`${sz} (${ratio}%)`);
  }
  const gz = gzipSize(tc.data);
  const gzRatio = (gz / tc.data.length * 100).toFixed(1);
  cols.push(`${gz} (${gzRatio}%)`);
  console.log('| ' + cols.join(' | ') + ' |');
}

console.log('\n> Uniform = no prediction (8 bits/symbol baseline)');
console.log('> Order-0 = current v0 backend (byte frequency tracking)');
console.log('> Order-1 = v1 candidate (previous-byte context, 256 contexts)');
console.log('> gzip -9 = reference (LZ77 + Huffman, maximum compression)');
