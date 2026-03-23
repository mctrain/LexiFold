#!/usr/bin/env python3
"""
Train a byte-level RWKV model for LexiFold text compression.
Architecture: RWKV v4 with 2 layers, hidden_size=128, byte-level vocab=256.
~500K params, ~2MB weights.

Usage: python3 tools/train_rwkv.py
Output: entry/src/main/resources/rawfile/rwkv_weights.bin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import os
import time
import math

# === Hyperparameters ===
VOCAB_SIZE = 256
HIDDEN_DIM = 128
FFN_DIM = 512
NUM_LAYERS = 2
SEQ_LEN = 256
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 40
GRAD_CLIP = 1.0

MAGIC = b'LXRW'
VERSION = 1

# === Larger training corpus ===
# ~100KB diverse text: English prose, code, JSON, logs, Chinese, etc.
CORPUS_PARTS = [
    # English literature
    """In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the
ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit
down on or to eat: it was a hobbit-hole, and that means comfort.
It had a perfectly round door like a porthole, painted green, with a shiny yellow brass knob
in the exact middle. The door opened on to a tube-shaped hall like a tunnel: a very comfortable
tunnel without smoke, with panelled walls, and floors tiled and carpeted, provided with polished
chairs, and lots and lots of pegs for hats and coats - the hobbit was fond of visitors.\n""" * 5,

    # English prose patterns
    """The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.
How vexingly quick daft zebras jump! The five boxing wizards jump quickly at dawn.
She sells sea shells by the sea shore. Peter Piper picked a peck of pickled peppers.
A stitch in time saves nine. All that glitters is not gold. Actions speak louder than words.
The early bird catches the worm. Better late than never. Every cloud has a silver lining.
Fortune favors the bold. Knowledge is power. Time flies when you're having fun.\n""" * 10,

    # Shakespeare-style
    """To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer
the slings and arrows of outrageous fortune, or to take arms against a sea of troubles,
and by opposing end them. To die, to sleep; to sleep, perchance to dream: ay, there's the rub,
for in that sleep of death what dreams may come, when we have shuffled off this mortal coil,
must give us pause. There's the respect that makes calamity of so long life.\n""" * 5,

    # Technical English
    """The compression algorithm works by predicting the next byte in a sequence. When the
prediction is accurate, fewer bits are needed to encode the actual value. The arithmetic
coder assigns shorter codes to higher-probability symbols and longer codes to lower-probability
ones. This process is mathematically optimal: the number of bits used approaches the Shannon
entropy of the source, given the model's probability estimates. The key to achieving good
compression ratios lies in the quality of the predictive model.\n""" * 5,

    # Source code (TypeScript/JavaScript patterns)
    """function fibonacci(n: number): number {
  if (n <= 1) return n;
  let a = 0, b = 1;
  for (let i = 2; i <= n; i++) {
    const temp = a + b;
    a = b; b = temp;
  }
  return b;
}

class LinkedList<T> {
  private head: ListNode<T> | null = null;
  private size: number = 0;

  push(value: T): void {
    const node = new ListNode(value);
    node.next = this.head;
    this.head = node;
    this.size++;
  }

  pop(): T | undefined {
    if (!this.head) return undefined;
    const value = this.head.value;
    this.head = this.head.next;
    this.size--;
    return value;
  }

  get length(): number { return this.size; }
}

async function fetchData(url: string): Promise<Response> {
  const response = await fetch(url, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response;
}

interface UserProfile {
  id: number;
  name: string;
  email: string;
  role: 'admin' | 'user' | 'guest';
  createdAt: Date;
  settings: Record<string, string>;
}

export function validateEmail(email: string): boolean {
  const regex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/;
  return regex.test(email);
}\n""" * 3,

    # JSON data patterns
    '\n'.join([
        f'{{"id":{i},"name":"user_{i}","email":"user{i}@example.com","score":{(i*17+3)%100},"active":{str(i%3!=0).lower()},"tags":["tag{i%5}","tag{(i+1)%5}"]}}'
        for i in range(200)
    ]) + '\n',

    # Log file patterns
    '\n'.join([
        f'2026-03-21 {8+i//60:02d}:{i%60:02d}:{(i*7)%60:02d}.{(i*13)%1000:03d} [{["INFO","DEBUG","WARN","ERROR"][i%4]}] {["auth","db","api","cache","scheduler"][i%5]}: {"Request processed in " + str((i*7+3)%500) + "ms" if i%3!=0 else "Connection timeout after 30000ms, retrying"}'
        for i in range(200)
    ]) + '\n',

    # HTML patterns
    """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LexiFold - Text Compression</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; }
    .container { max-width: 800px; margin: 0 auto; }
    .card { background: #fff; border-radius: 12px; padding: 16px; margin: 12px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }
    .btn { padding: 8px 16px; border-radius: 8px; border: none; cursor: pointer; font-size: 14px; }
    .btn-primary { background: #0a59f7; color: white; }
  </style>
</head>
<body>
  <div class="container">
    <h1>LexiFold Compression</h1>
    <div class="card">
      <textarea id="input" rows="6" placeholder="Enter text to compress..."></textarea>
      <button class="btn btn-primary" onclick="compress()">Compress</button>
    </div>
  </div>
</body>
</html>\n""" * 3,

    # CSV patterns
    'id,name,email,department,salary,start_date,active\n' +
    '\n'.join([
        f'{i},{["Alice","Bob","Charlie","Diana","Eve","Frank"][i%6]}_{i},{["alice","bob","charlie","diana","eve","frank"][i%6]}{i}@corp.com,{["Engineering","Marketing","Sales","HR","Finance"][i%5]},{50000+(i*1234)%50000},{2020+i%6}-{1+i%12:02d}-{1+i%28:02d},{str(i%4!=0).lower()}'
        for i in range(200)
    ]) + '\n',

    # Markdown
    """# LexiFold Architecture

## Overview

LexiFold compresses text using **predictive modeling** combined with **arithmetic coding**.

### How it works

1. The predictor estimates probability distribution over the next byte
2. The arithmetic coder uses this distribution to encode efficiently
3. Higher prediction accuracy → lower entropy → better compression

### Supported backends

| Backend | Type | Compression |
|---------|------|-------------|
| Order-0 | Statistical | ~62% |
| Order-1 | Statistical | ~43% |
| PPM-3 | Statistical | ~11% |
| RWKV | Neural | TBD |

## Installation

```bash
git clone https://github.com/mctrain/LexiFold.git
cd LexiFold
# Build with DevEco Studio or hvigorw
```
\n""" * 3,

    # Chinese text
    """今天天气真好，适合出去散步。春风拂面，花香四溢，令人心旷神怡。
这是一段用于测试中文文本压缩效果的示例内容，包含常见的中文字符和标点符号。
深度学习模型能够通过学习文本中的统计规律来预测下一个字节，从而实现高效压缩。
人工智能技术正在改变我们处理和存储数据的方式，文本压缩是其中的一个重要应用场景。
神经网络通过隐藏状态记忆上下文信息，比传统的统计模型具有更强的预测能力。\n""" * 8,
]

# === RWKV v4 Model ===

class RWKV_TimeMix(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_decay = nn.Parameter(torch.randn(hidden_dim) * 0.1 - 1.0)
        self.time_first = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.time_mix_k = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.receptance = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, state):
        # state: (state_x, state_A, state_B, state_p)
        state_x, state_A, state_B, state_p = state
        xk = x * self.time_mix_k + state_x * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + state_x * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + state_x * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        ww = self.time_first + k
        p = torch.maximum(state_p, ww)
        e1 = torch.exp(state_p - p)
        e2 = torch.exp(ww - p)
        a = e1 * state_A + e2 * v
        b = e1 * state_B + e2

        wkv = a / (b + 1e-8)
        rwkv = r * wkv

        # Update state
        ww2 = state_p + self.time_decay
        p2 = torch.maximum(ww2, k)
        e1 = torch.exp(ww2 - p2)
        e2 = torch.exp(k - p2)
        new_A = e1 * state_A + e2 * v
        new_B = e1 * state_B + e2
        new_state = (x.detach(), new_A.detach(), new_B.detach(), p2.detach())

        return self.output(rwkv), new_state


class RWKV_ChannelMix(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.time_mix_k = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.key = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.value = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.receptance = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, state_ffn):
        xk = x * self.time_mix_k + state_ffn * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + state_ffn * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(xr))
        return r * kv, x.detach()


class RWKV_Block(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.time_mix = RWKV_TimeMix(hidden_dim)
        self.channel_mix = RWKV_ChannelMix(hidden_dim, ffn_dim)

    def forward(self, x, tm_state, cm_state):
        h = self.ln1(x)
        att, new_tm_state = self.time_mix(h, tm_state)
        x = x + att
        h = self.ln2(x)
        ffn, new_cm_state = self.channel_mix(h, cm_state)
        x = x + ffn
        return x, new_tm_state, new_cm_state


class RWKV_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ln0 = nn.LayerNorm(HIDDEN_DIM)
        self.blocks = nn.ModuleList([RWKV_Block(HIDDEN_DIM, FFN_DIM) for _ in range(NUM_LAYERS)])
        self.ln_out = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)
        nn.init.uniform_(self.emb.weight, -0.01, 0.01)
        nn.init.xavier_uniform_(self.head.weight)

    def init_state(self, batch_size=1):
        """Initialize per-layer states."""
        states = []
        for _ in range(NUM_LAYERS):
            tm = (
                torch.zeros(batch_size, HIDDEN_DIM),
                torch.zeros(batch_size, HIDDEN_DIM),
                torch.zeros(batch_size, HIDDEN_DIM),
                torch.full((batch_size, HIDDEN_DIM), -1e30),
            )
            cm = torch.zeros(batch_size, HIDDEN_DIM)
            states.append((tm, cm))
        return states

    def forward_step(self, token_id, states):
        x = self.emb(token_id)
        x = self.ln0(x)
        new_states = []
        for i, block in enumerate(self.blocks):
            tm_state, cm_state = states[i]
            x, new_tm, new_cm = block(x, tm_state, cm_state)
            new_states.append((new_tm, new_cm))
        x = self.ln_out(x)
        logits = self.head(x)
        return logits, new_states

    def forward_sequence(self, tokens):
        """Process a full sequence, return logits for each position."""
        B, T = tokens.shape
        states = self.init_state(B)
        all_logits = []
        for t in range(T):
            logits, states = self.forward_step(tokens[:, t], states)
            all_logits.append(logits)
        return torch.stack(all_logits, dim=1)

    def save_weights(self, path):
        state = self.state_dict()
        with open(path, 'wb') as f:
            f.write(MAGIC)
            f.write(struct.pack('<III', VERSION, HIDDEN_DIM, NUM_LAYERS))
            f.write(struct.pack('<II', FFN_DIM, VOCAB_SIZE))
            # Write all parameters in state_dict order
            keys = sorted(state.keys())
            f.write(struct.pack('<I', len(keys)))
            for key in keys:
                # Write key name
                key_bytes = key.encode('utf-8')
                f.write(struct.pack('<I', len(key_bytes)))
                f.write(key_bytes)
                # Write tensor
                tensor = state[key].float().contiguous()
                ndim = len(tensor.shape)
                f.write(struct.pack('<I', ndim))
                for s in tensor.shape:
                    f.write(struct.pack('<I', s))
                f.write(tensor.numpy().tobytes())
        total_params = sum(p.numel() for p in self.parameters())
        size = os.path.getsize(path)
        print(f"Saved {len(keys)} tensors, {total_params:,} params, {size:,} bytes to {path}")


def prepare_data(corpus):
    data = torch.tensor(list(corpus.encode('utf-8')), dtype=torch.long)
    n = len(data)
    sequences = []
    stride = SEQ_LEN // 2
    for i in range(0, n - SEQ_LEN - 1, stride):
        sequences.append(data[i:i + SEQ_LEN + 1])
    return torch.stack(sequences)


def main():
    print("=== LexiFold RWKV Training ===\n")
    print(f"Model: RWKV v4, {NUM_LAYERS} layers, hidden={HIDDEN_DIM}, FFN={FFN_DIM}, vocab={VOCAB_SIZE}")

    device = 'cpu'
    model = RWKV_Model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params * 4 / 1024:.1f} KB)")

    corpus = ''.join(CORPUS_PARTS)
    print(f"Corpus size: {len(corpus):,} chars ({len(corpus.encode('utf-8')):,} bytes UTF-8)")

    dataset = prepare_data(corpus)
    print(f"Training sequences: {len(dataset)} (seq_len={SEQ_LEN})\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(dataset))
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for start in range(0, len(dataset), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(dataset))
            idx = perm[start:end]
            batch = dataset[idx].to(device)  # (B, SEQ_LEN+1)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model.forward_sequence(inputs)  # (B, SEQ_LEN, VOCAB_SIZE)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        bpc = avg_loss / math.log(2)
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  loss={avg_loss:.4f}  bpc={bpc:.3f}  lr={lr:.6f}  time={elapsed:.1f}s")

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'entry', 'src', 'main', 'resources', 'rawfile')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'rwkv_weights.bin')
    model.save_weights(output_path)

    # Eval
    print("\n=== Evaluation ===")
    model.eval()
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        '{"name":"test","value":42,"active":true}',
        "function hello() { return 'world'; }",
        "2026-03-21 08:00:01 [INFO] server: OK",
    ]
    with torch.no_grad():
        for text in test_texts:
            data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long).unsqueeze(0)
            states = model.init_state(1)
            total_bits = 0.0
            for t in range(data.shape[1] - 1):
                logits, states = model.forward_step(data[:, t], states)
                probs = F.softmax(logits, dim=-1)
                target = data[0, t + 1].item()
                total_bits -= math.log2(probs[0, target].item() + 1e-10)
            bpc = total_bits / (data.shape[1] - 1)
            ratio = bpc / 8 * 100
            print(f"  '{text[:50]}...'  bpc={bpc:.3f}  ratio={ratio:.1f}%")

    print("\nDone!")


if __name__ == '__main__':
    main()
