// Benchmark: compare compression across backends, preprocessors, and reference compressors
// Run: npx tsx tools/benchmark.ts

import { execSync } from 'child_process';
import { writeFileSync, unlinkSync, statSync, readFileSync } from 'fs';

// === Constants (mirror from core) ===
const CODE_BITS = 24;
const TOP_VALUE = 1 << CODE_BITS;
const HALF = 1 << (CODE_BITS - 1);
const FIRST_QTR = 1 << (CODE_BITS - 2);
const THIRD_QTR = HALF + FIRST_QTR;
const BYTE_VOCAB_SIZE = 256;
const MAX_TOTAL_FREQ = 16384;
const ESCAPE_CHAR = '\u0001';

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
  name = 'Order-0';
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
  name = 'Order-1';
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

class PPMOrder3Blended implements Backend {
  name = 'PPM-3 Blended';
  private o0Counts = new Array(256).fill(1);
  private o0Total = 256;
  private o1Counts: number[][] = [];
  private o1Totals: number[] = [];
  private o2Contexts = new Map<number, { counts: number[], total: number }>();
  private o3Contexts = new Map<number, { counts: number[], total: number }>();
  private prev1 = 0;
  private prev2 = 0;
  private prev3 = 0;
  private blended = new Array(256).fill(1);
  private blendedTotal = 256;

  reset() {
    this.o0Counts = new Array(256).fill(1);
    this.o0Total = 256;
    this.o1Counts = [];
    this.o1Totals = [];
    for (let i = 0; i < 256; i++) {
      this.o1Counts.push(new Array(256).fill(1));
      this.o1Totals.push(256);
    }
    this.o2Contexts.clear();
    this.o3Contexts.clear();
    this.prev1 = 0; this.prev2 = 0; this.prev3 = 0;
    this.blended = new Array(256).fill(1);
    this.blendedTotal = 256;
  }

  private recomputeBlended() {
    const o1t = this.o1Totals[this.prev1];
    const o1c = this.o1Counts[this.prev1];
    const o2key = this.prev2 * 256 + this.prev1;
    const o2 = this.o2Contexts.get(o2key);
    const o3key = this.prev3 * 65536 + this.prev2 * 256 + this.prev1;
    const o3 = this.o3Contexts.get(o3key);

    const w0 = 1;
    const w1 = Math.min(Math.max(1, Math.floor(o1t / 4)), 16);
    const w2 = o2 && o2.total > 0 ? Math.min(Math.max(1, Math.floor(o2.total / 2)), 32) : 0;
    const w3 = o3 && o3.total > 0 ? Math.min(Math.max(1, o3.total), 64) : 0;
    const wsum = w0 + w1 + w2 + w3;
    const SCALE = 4096;

    this.blendedTotal = 0;
    for (let s = 0; s < 256; s++) {
      let score = w0 * this.o0Counts[s] * SCALE / this.o0Total;
      if (o1t > 0) score += w1 * o1c[s] * SCALE / o1t;
      if (o2 && o2.total > 0) score += w2 * o2.counts[s] * SCALE / o2.total;
      if (o3 && o3.total > 0) score += w3 * o3.counts[s] * SCALE / o3.total;
      const count = Math.max(1, Math.floor(score / wsum));
      this.blended[s] = count;
      this.blendedTotal += count;
    }
    if (this.blendedTotal > MAX_TOTAL_FREQ) {
      this.blendedTotal = 0;
      for (let i = 0; i < 256; i++) {
        this.blended[i] = Math.max(1, Math.floor(this.blended[i] / 2));
        this.blendedTotal += this.blended[i];
      }
    }
  }

  getFrequency(sym: number) {
    let cumLow = 0;
    for (let i = 0; i < sym; i++) cumLow += this.blended[i];
    return { cumLow, cumHigh: cumLow + this.blended[sym] };
  }
  getTotal() { return this.blendedTotal; }
  update(sym: number) {
    // Update order-0
    this.o0Counts[sym]++; this.o0Total++;
    if (this.o0Total >= MAX_TOTAL_FREQ) {
      this.o0Total = 0;
      for (let i = 0; i < 256; i++) { this.o0Counts[i] = Math.max(1, Math.floor(this.o0Counts[i] / 2)); this.o0Total += this.o0Counts[i]; }
    }
    // Update order-1
    this.o1Counts[this.prev1][sym]++; this.o1Totals[this.prev1]++;
    if (this.o1Totals[this.prev1] >= MAX_TOTAL_FREQ) {
      this.o1Totals[this.prev1] = 0;
      for (let i = 0; i < 256; i++) { this.o1Counts[this.prev1][i] = Math.max(1, Math.floor(this.o1Counts[this.prev1][i] / 2)); this.o1Totals[this.prev1] += this.o1Counts[this.prev1][i]; }
    }
    // Update order-2
    const o2key = this.prev2 * 256 + this.prev1;
    let o2 = this.o2Contexts.get(o2key);
    if (!o2) { o2 = { counts: new Array(256).fill(0), total: 0 }; this.o2Contexts.set(o2key, o2); }
    o2.counts[sym]++; o2.total++;
    if (o2.total >= MAX_TOTAL_FREQ) {
      o2.total = 0;
      for (let i = 0; i < 256; i++) { o2.counts[i] = Math.max(o2.counts[i] > 0 ? 1 : 0, Math.floor(o2.counts[i] / 2)); o2.total += o2.counts[i]; }
    }
    // Update order-3
    const o3key = this.prev3 * 65536 + this.prev2 * 256 + this.prev1;
    let o3 = this.o3Contexts.get(o3key);
    if (!o3) { o3 = { counts: new Array(256).fill(0), total: 0 }; this.o3Contexts.set(o3key, o3); }
    o3.counts[sym]++; o3.total++;
    if (o3.total >= MAX_TOTAL_FREQ) {
      o3.total = 0;
      for (let i = 0; i < 256; i++) { o3.counts[i] = Math.max(o3.counts[i] > 0 ? 1 : 0, Math.floor(o3.counts[i] / 2)); o3.total += o3.counts[i]; }
    }
    // Shift context
    this.prev3 = this.prev2; this.prev2 = this.prev1; this.prev1 = sym;
    this.recomputeBlended();
  }
}

// === Preprocessors ===
function lowercasePreprocess(text: string): string {
  let result = '';
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === ESCAPE_CHAR) { result += ESCAPE_CHAR + ESCAPE_CHAR; }
    else if (ch >= 'A' && ch <= 'Z') { result += ESCAPE_CHAR + ch.toLowerCase(); }
    else { result += ch; }
  }
  return result;
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

// === Test data generators ===
const enc = new TextEncoder();

// 1. English prose (repetitive)
let english = '';
for (let i = 0; i < 100; i++) english += `Line ${i}: The quick brown fox jumps over the lazy dog.\n`;
const englishData = enc.encode(english);

// 2. Short English sentence
const shortText = enc.encode('The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump!');

// 3. JSON lines (deterministic)
let jsonStr = '';
let jseed = 1234;
for (let i = 0; i < 50; i++) {
  jseed = (jseed * 1103515245 + 12345) & 0x7FFFFFFF;
  jsonStr += `{"id":${i},"name":"item_${i}","value":${(jseed >> 16) % 1000},"active":${i % 2 === 0}}\n`;
}
const jsonData = enc.encode(jsonStr);

// 4. Chinese text
const chineseText = '今天天气真好，适合出去散步。春风拂面，花香四溢，令人心旷神怡。这是一段用于测试中文文本压缩效果的示例内容，包含常见的中文字符和标点符号。';
let chineseLong = '';
for (let i = 0; i < 20; i++) chineseLong += chineseText;
const chineseData = enc.encode(chineseLong);

// 5. Source code (TypeScript)
let sourceCode = '';
for (let i = 0; i < 30; i++) {
  sourceCode += `function calc_${i}(a: number, b: number): number {\n`;
  sourceCode += `  const result = a * ${i} + b;\n`;
  sourceCode += `  if (result > 0) { return result; }\n`;
  sourceCode += `  return -result;\n}\n\n`;
}
const sourceData = enc.encode(sourceCode);

// 6. Random bytes
const random = new Uint8Array(4096);
let seed = 42;
for (let i = 0; i < 4096; i++) { seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF; random[i] = (seed >> 16) & 0xFF; }

// 7. HTML markup
let html = '<!DOCTYPE html><html><head><title>Test Page</title></head><body>\n';
for (let i = 0; i < 30; i++) {
  html += `  <div class="item" id="item-${i}"><span class="label">Item ${i}</span><span class="value">${i * 17}</span></div>\n`;
}
html += '</body></html>\n';
const htmlData = enc.encode(html);

// 8. CSV data
let csv = 'id,name,email,score,grade,active\n';
const names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Hank'];
const domains = ['example.com', 'test.org', 'demo.net'];
let cseed = 99;
for (let i = 0; i < 100; i++) {
  cseed = (cseed * 1103515245 + 12345) & 0x7FFFFFFF;
  const name = names[i % names.length];
  const domain = domains[i % domains.length];
  const score = (cseed >> 16) % 100;
  const grade = ['A', 'B', 'C', 'D', 'F'][Math.floor(score / 20)];
  csv += `${i},${name}_${i},${name.toLowerCase()}${i}@${domain},${score},${grade},${i % 3 !== 0}\n`;
}
const csvData = enc.encode(csv);

// 9. Log file
let logStr = '';
for (let i = 0; i < 80; i++) {
  const levels = ['INFO', 'DEBUG', 'WARN', 'ERROR', 'INFO', 'INFO', 'DEBUG', 'INFO'];
  const modules = ['auth', 'db', 'api', 'cache', 'scheduler'];
  logStr += `2026-03-21 08:${String(i % 60).padStart(2, '0')}:${String(i % 60).padStart(2, '0')}.${String(i * 13 % 1000).padStart(3, '0')} [${levels[i % levels.length]}] ${modules[i % modules.length]}: `;
  if (i % 7 === 0) logStr += `Connection timeout after 30000ms, retrying (attempt ${i % 5 + 1}/5)\n`;
  else if (i % 5 === 0) logStr += `Cache miss for key user:${i}, fetching from database\n`;
  else logStr += `Request processed successfully in ${(i * 7 + 3) % 500}ms\n`;
}
const logData = enc.encode(logStr);

// 10. Mixed multilingual
let mixed = '';
for (let i = 0; i < 20; i++) {
  mixed += `Entry ${i}: Hello World! 你好世界！こんにちは世界！Привет мир! مرحبا بالعالم\n`;
  mixed += `  Score: ${i * 17 % 100}, Status: ${i % 2 === 0 ? 'active' : 'inactive'}\n`;
}
const mixedData = enc.encode(mixed);

interface TestCase { name: string; data: Uint8Array; }

const testCases: TestCase[] = [
  { name: 'English prose (5.4KB)', data: englishData },
  { name: 'Short English (122B)', data: shortText },
  { name: 'JSON lines (3.0KB)', data: jsonData },
  { name: 'Chinese text (8.7KB)', data: chineseData },
  { name: 'TypeScript code (3.8KB)', data: sourceData },
  { name: 'HTML markup (2.2KB)', data: htmlData },
  { name: 'CSV data (5.6KB)', data: csvData },
  { name: 'Log file (5.5KB)', data: logData },
  { name: 'Multilingual (2.8KB)', data: mixedData },
  { name: 'Random bytes (4KB)', data: random },
];

interface Config {
  label: string;
  backend: Backend;
  preprocess?: (text: string) => string;
}

const configs: Config[] = [
  { label: 'v0 Order-0', backend: new Order0Adaptive() },
  { label: 'v1 Order-1', backend: new Order1Adaptive() },
  { label: 'v2 PPM-3', backend: new PPMOrder3Blended() },
];

// === Run benchmark ===
console.log('# LexiFold Compression Benchmark\n');
console.log(`Date: ${new Date().toISOString().slice(0, 10)}\n`);

// Header
const hdr = ['Test Data', 'Size', ...configs.map(c => c.label), 'gzip -9'];
console.log('| ' + hdr.join(' | ') + ' |');
console.log('|' + hdr.map(() => ' --- ').join('|') + '|');

for (const tc of testCases) {
  const cols: string[] = [tc.name, `${tc.data.length}`];

  for (const cfg of configs) {
    let data = tc.data;
    if (cfg.preprocess) {
      // Preprocess: decode to string, preprocess, re-encode
      const text = new TextDecoder().decode(tc.data);
      const processed = cfg.preprocess(text);
      data = enc.encode(processed);
    }
    const sz = compressSize(data, cfg.backend);
    const ratio = (sz / tc.data.length * 100).toFixed(1);
    cols.push(`${ratio}%`);
  }

  const gz = gzipSize(tc.data);
  const gzRatio = (gz / tc.data.length * 100).toFixed(1);
  cols.push(`${gzRatio}%`);

  console.log('| ' + cols.join(' | ') + ' |');
}

console.log('');
console.log('**Legend:**');
console.log('- **v0 Order-0**: Byte frequency tracking (v0 default)');
console.log('- **v1 Order-1**: Previous-byte context model (v1 default)');
console.log('- **v2 PPM-3**: Blended order-0/1/2/3 context model, native C++ engine (v2)');
console.log('- **gzip -9**: Reference (LZ77 + Huffman, maximum compression)');
