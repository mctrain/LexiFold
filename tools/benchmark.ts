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

// === GRU Neural Backend (TS replica of C++ engine) ===
class GRUBackend implements Backend {
  name = 'GRU Neural';
  private embed: Float32Array = new Float32Array(0);
  private W_z: Float32Array = new Float32Array(0);
  private b_z: Float32Array = new Float32Array(0);
  private W_r: Float32Array = new Float32Array(0);
  private b_r: Float32Array = new Float32Array(0);
  private W_h: Float32Array = new Float32Array(0);
  private b_h: Float32Array = new Float32Array(0);
  private out_w: Float32Array = new Float32Array(0);
  private out_b: Float32Array = new Float32Array(0);
  private hidden: Float32Array = new Float32Array(0);
  private freqTable: number[] = new Array(256).fill(1);
  private freqTotal = 256;
  private embedDim = 0;
  private hiddenDim = 0;
  private loaded = false;

  constructor(weightsPath: string) {
    this.loadWeights(weightsPath);
    this.reset();
  }

  private loadWeights(path: string): void {
    const buf = readFileSync(path);
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    // Header: magic(4) + version(4) + embed_dim(4) + hidden_dim(4) + vocab(4) = 20 bytes
    const magic = dv.getUint32(0, true);
    if (magic !== 0x4E4E584C) { console.error('Bad GRU magic'); return; }
    this.embedDim = dv.getUint32(8, true);
    this.hiddenDim = dv.getUint32(12, true);
    const inputSize = this.embedDim + this.hiddenDim;
    let off = 20;
    const readF32 = (count: number): Float32Array => {
      const arr = new Float32Array(count);
      for (let i = 0; i < count; i++) { arr[i] = dv.getFloat32(off, true); off += 4; }
      return arr;
    };
    this.embed = readF32(256 * this.embedDim);
    this.W_z = readF32(inputSize * this.hiddenDim);
    this.b_z = readF32(this.hiddenDim);
    this.W_r = readF32(inputSize * this.hiddenDim);
    this.b_r = readF32(this.hiddenDim);
    this.W_h = readF32(inputSize * this.hiddenDim);
    this.b_h = readF32(this.hiddenDim);
    this.out_w = readF32(this.hiddenDim * 256);
    this.out_b = readF32(256);
    this.hidden = new Float32Array(this.hiddenDim);
    this.loaded = true;
  }

  reset() {
    if (this.hiddenDim > 0) this.hidden.fill(0);
    this.freqTable = new Array(256).fill(1);
    this.freqTotal = 256;
  }

  private sigmoid(x: number): number { return 1.0 / (1.0 + Math.exp(-Math.max(-15, Math.min(15, x)))); }

  private forward(byte: number): void {
    if (!this.loaded) return;
    const ed = this.embedDim, hd = this.hiddenDim, is = ed + hd;
    // xh = [embed[byte], hidden]
    const xh = new Float32Array(is);
    for (let i = 0; i < ed; i++) xh[i] = this.embed[byte * ed + i];
    for (let i = 0; i < hd; i++) xh[ed + i] = this.hidden[i];
    // z = sigmoid(xh @ W_z + b_z)
    const z = new Float32Array(hd);
    for (let j = 0; j < hd; j++) {
      let s = this.b_z[j];
      for (let i = 0; i < is; i++) s += xh[i] * this.W_z[i * hd + j];
      z[j] = this.sigmoid(s);
    }
    // r = sigmoid(xh @ W_r + b_r)
    const r = new Float32Array(hd);
    for (let j = 0; j < hd; j++) {
      let s = this.b_r[j];
      for (let i = 0; i < is; i++) s += xh[i] * this.W_r[i * hd + j];
      r[j] = this.sigmoid(s);
    }
    // xrh = [embed, r*h]
    const xrh = new Float32Array(is);
    for (let i = 0; i < ed; i++) xrh[i] = xh[i];
    for (let i = 0; i < hd; i++) xrh[ed + i] = r[i] * this.hidden[i];
    // h_cand = tanh(xrh @ W_h + b_h)
    const hc = new Float32Array(hd);
    for (let j = 0; j < hd; j++) {
      let s = this.b_h[j];
      for (let i = 0; i < is; i++) s += xrh[i] * this.W_h[i * hd + j];
      hc[j] = Math.tanh(Math.max(-15, Math.min(15, s)));
    }
    // h_new = (1-z)*h + z*h_cand
    for (let i = 0; i < hd; i++) this.hidden[i] = (1 - z[i]) * this.hidden[i] + z[i] * hc[i];
    // logits = h @ out_w + out_b
    const logits = new Float32Array(256);
    for (let j = 0; j < 256; j++) {
      let s = this.out_b[j];
      for (let i = 0; i < hd; i++) s += this.hidden[i] * this.out_w[i * 256 + j];
      logits[j] = s;
    }
    // softmax → freq table
    let maxL = -Infinity;
    for (let i = 0; i < 256; i++) if (logits[i] > maxL) maxL = logits[i];
    let sumExp = 0;
    const probs = new Float32Array(256);
    for (let i = 0; i < 256; i++) { probs[i] = Math.exp(logits[i] - maxL); sumExp += probs[i]; }
    const target = 8192;
    this.freqTotal = 0;
    for (let i = 0; i < 256; i++) {
      const c = Math.max(1, Math.floor(probs[i] / sumExp * target));
      this.freqTable[i] = c;
      this.freqTotal += c;
    }
  }

  getFrequency(sym: number) {
    let cumLow = 0;
    for (let i = 0; i < sym; i++) cumLow += this.freqTable[i];
    return { cumLow, cumHigh: cumLow + this.freqTable[sym] };
  }
  getTotal() { return this.freqTotal; }
  update(sym: number) { this.forward(sym); }
}

// === RWKV Backend (TS replica of C++ engine) ===
class RWKVBackend implements Backend {
  name = 'RWKV';
  private weights: Map<string, { shape: number[], data: Float32Array }> = new Map();
  private hiddenDim = 0;
  private ffnDim = 0;
  private numLayers = 0;
  private states: { sx: Float32Array, sA: Float32Array, sB: Float32Array, sp: Float32Array, sf: Float32Array }[] = [];
  private freqTable: number[] = new Array(256).fill(1);
  private freqTotal = 256;
  private loaded = false;

  constructor(weightsPath: string) {
    const buf = readFileSync(weightsPath);
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    let off = 0;
    off += 4; // magic
    off += 4; // version
    this.hiddenDim = dv.getUint32(off, true); off += 4;
    this.numLayers = dv.getUint32(off, true); off += 4;
    this.ffnDim = dv.getUint32(off, true); off += 4;
    off += 4; // vocab
    const numTensors = dv.getUint32(off, true); off += 4;
    for (let t = 0; t < numTensors; t++) {
      const keyLen = dv.getUint32(off, true); off += 4;
      const key = new TextDecoder().decode(buf.slice(off, off + keyLen)); off += keyLen;
      const ndim = dv.getUint32(off, true); off += 4;
      const shape: number[] = [];
      let total = 1;
      for (let d = 0; d < ndim; d++) { const s = dv.getUint32(off, true); off += 4; shape.push(s); total *= s; }
      const data = new Float32Array(total);
      for (let i = 0; i < total; i++) { data[i] = dv.getFloat32(off, true); off += 4; }
      this.weights.set(key, { shape, data });
    }
    this.loaded = true;
    this.reset();
  }

  private w(name: string): Float32Array { return this.weights.get(name)!.data; }

  private layerNorm(x: Float32Array, prefix: string): void {
    const wt = this.w(prefix + '.weight'), bi = this.w(prefix + '.bias');
    const n = x.length;
    let mean = 0; for (let i = 0; i < n; i++) mean += x[i]; mean /= n;
    let vr = 0; for (let i = 0; i < n; i++) { const d = x[i] - mean; vr += d * d; } vr /= n;
    const inv = 1.0 / Math.sqrt(vr + 1e-5);
    for (let i = 0; i < n; i++) x[i] = (x[i] - mean) * inv * wt[i] + bi[i];
  }

  private matVec(W: Float32Array, x: Float32Array, rows: number, cols: number): Float32Array {
    const out = new Float32Array(rows);
    for (let r = 0; r < rows; r++) { let s = 0; for (let c = 0; c < cols; c++) s += W[r * cols + c] * x[c]; out[r] = s; }
    return out;
  }

  private sigmoid(x: number): number { return 1.0 / (1.0 + Math.exp(-Math.max(-15, Math.min(15, x)))); }

  reset() {
    this.states = [];
    for (let i = 0; i < this.numLayers; i++) {
      this.states.push({
        sx: new Float32Array(this.hiddenDim),
        sA: new Float32Array(this.hiddenDim),
        sB: new Float32Array(this.hiddenDim),
        sp: new Float32Array(this.hiddenDim).fill(-1e30),
        sf: new Float32Array(this.hiddenDim),
      });
    }
    this.freqTable = new Array(256).fill(1); this.freqTotal = 256;
  }

  private forward(byte: number): void {
    if (!this.loaded) return;
    const H = this.hiddenDim, F = this.ffnDim;
    const emb = this.w('emb.weight');
    let x = new Float32Array(H);
    for (let i = 0; i < H; i++) x[i] = emb[byte * H + i];
    this.layerNorm(x, 'ln0');

    for (let layer = 0; layer < this.numLayers; layer++) {
      const p = `blocks.${layer}`;
      const st = this.states[layer];
      const xRes = new Float32Array(x);

      // TimeMix
      this.layerNorm(x, `${p}.ln1`);
      const mk = this.w(`${p}.time_mix.time_mix_k`), mv = this.w(`${p}.time_mix.time_mix_v`), mr = this.w(`${p}.time_mix.time_mix_r`);
      const xk = new Float32Array(H), xv = new Float32Array(H), xr = new Float32Array(H);
      for (let i = 0; i < H; i++) {
        xk[i] = x[i] * mk[i] + st.sx[i] * (1 - mk[i]);
        xv[i] = x[i] * mv[i] + st.sx[i] * (1 - mv[i]);
        xr[i] = x[i] * mr[i] + st.sx[i] * (1 - mr[i]);
      }
      const k = this.matVec(this.w(`${p}.time_mix.key.weight`), xk, H, H);
      const v = this.matVec(this.w(`${p}.time_mix.value.weight`), xv, H, H);
      const r = this.matVec(this.w(`${p}.time_mix.receptance.weight`), xr, H, H);
      for (let i = 0; i < H; i++) r[i] = this.sigmoid(r[i]);

      const tf = this.w(`${p}.time_mix.time_first`), td = this.w(`${p}.time_mix.time_decay`);
      const wkv = new Float32Array(H);
      for (let i = 0; i < H; i++) {
        const ww = tf[i] + k[i];
        const pp = Math.max(st.sp[i], ww);
        const e1 = Math.exp(st.sp[i] - pp), e2 = Math.exp(ww - pp);
        const a = e1 * st.sA[i] + e2 * v[i], b = e1 * st.sB[i] + e2;
        wkv[i] = r[i] * a / (b + 1e-8);
      }
      const attOut = this.matVec(this.w(`${p}.time_mix.output.weight`), wkv, H, H);
      // Update state
      for (let i = 0; i < H; i++) {
        const ww2 = st.sp[i] + td[i], pp2 = Math.max(ww2, k[i]);
        const e1 = Math.exp(ww2 - pp2), e2 = Math.exp(k[i] - pp2);
        st.sA[i] = e1 * st.sA[i] + e2 * v[i]; st.sB[i] = e1 * st.sB[i] + e2; st.sp[i] = pp2;
      }
      st.sx = new Float32Array(x);

      x = new Float32Array(xRes);
      for (let i = 0; i < H; i++) x[i] += attOut[i];

      // ChannelMix
      const xRes2 = new Float32Array(x);
      this.layerNorm(x, `${p}.ln2`);
      const cmk = this.w(`${p}.channel_mix.time_mix_k`), cmr = this.w(`${p}.channel_mix.time_mix_r`);
      const ck = new Float32Array(H), cr = new Float32Array(H);
      for (let i = 0; i < H; i++) {
        ck[i] = x[i] * cmk[i] + st.sf[i] * (1 - cmk[i]);
        cr[i] = x[i] * cmr[i] + st.sf[i] * (1 - cmr[i]);
      }
      st.sf = new Float32Array(x);

      let fk = this.matVec(this.w(`${p}.channel_mix.key.weight`), ck, F, H);
      for (let i = 0; i < F; i++) { fk[i] = Math.max(0, fk[i]); fk[i] *= fk[i]; }
      const fv = this.matVec(this.w(`${p}.channel_mix.value.weight`), fk, H, F);
      const fr = this.matVec(this.w(`${p}.channel_mix.receptance.weight`), cr, H, H);
      for (let i = 0; i < H; i++) fv[i] *= this.sigmoid(fr[i]);

      x = new Float32Array(xRes2);
      for (let i = 0; i < H; i++) x[i] += fv[i];
    }

    this.layerNorm(x, 'ln_out');
    const logits = this.matVec(this.w('head.weight'), x, 256, this.hiddenDim);

    // softmax → freq table
    let maxL = -Infinity;
    for (let i = 0; i < 256; i++) if (logits[i] > maxL) maxL = logits[i];
    let sumExp = 0;
    const probs = new Float32Array(256);
    for (let i = 0; i < 256; i++) { probs[i] = Math.exp(logits[i] - maxL); sumExp += probs[i]; }
    this.freqTotal = 0;
    for (let i = 0; i < 256; i++) {
      const c = Math.max(1, Math.floor(probs[i] / sumExp * 8192));
      this.freqTable[i] = c; this.freqTotal += c;
    }
  }

  getFrequency(sym: number) {
    let cumLow = 0; for (let i = 0; i < sym; i++) cumLow += this.freqTable[i];
    return { cumLow, cumHigh: cumLow + this.freqTable[sym] };
  }
  getTotal() { return this.freqTotal; }
  update(sym: number) { this.forward(sym); }
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

const gruPath = './entry/src/main/resources/rawfile/gru_weights.bin';
const rwkvPath = './entry/src/main/resources/rawfile/rwkv_weights.bin';
let gruBackend: Backend | null = null;
let rwkvBackend: Backend | null = null;
try { gruBackend = new GRUBackend(gruPath); } catch (e) { /* skip */ }
try { rwkvBackend = new RWKVBackend(rwkvPath); } catch (e) { /* skip */ }

const configs: Config[] = [
  { label: 'v0 Order-0', backend: new Order0Adaptive() },
  { label: 'v1 Order-1', backend: new Order1Adaptive() },
  { label: 'v2 PPM-3', backend: new PPMOrder3Blended() },
];
if (gruBackend) configs.push({ label: 'v2 GRU', backend: gruBackend });
if (rwkvBackend) configs.push({ label: 'v2 RWKV', backend: rwkvBackend });

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
console.log('- **v2 PPM-3**: Blended order-0/1/2/3 context model, native C++ engine');
console.log('- **v2 GRU**: Character-level GRU (36K params, 7KB corpus)');
console.log('- **v2 RWKV**: RWKV v4 (495K params, 74KB corpus)');
console.log('- **gzip -9**: Reference (LZ77 + Huffman, maximum compression)');
