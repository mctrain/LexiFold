#!/usr/bin/env python3
"""
Train a tiny character-level GRU for LexiFold text compression.
Architecture: Embedding(256→16) → GRU(16→64) → Linear(64→256)
~36K params, ~146KB weights.

Usage: python3 tools/train_gru.py
Output: entry/src/main/resources/rawfile/gru_weights.bin
"""

import numpy as np
import struct
import os
import time

# === Hyperparameters ===
VOCAB_SIZE = 256
EMBED_DIM = 16
HIDDEN_DIM = 64
SEQ_LEN = 64        # training sequence length
BATCH_SIZE = 32
LEARNING_RATE = 0.003
EPOCHS = 30
CLIP_GRAD = 1.0

# === Weight file format ===
# Magic: b'LXNN' (4 bytes)
# Version: uint32 (4 bytes)
# embed_dim: uint32 (4 bytes)
# hidden_dim: uint32 (4 bytes)
# vocab_size: uint32 (4 bytes)
# Then float32 arrays in order:
#   embed_weight: [vocab_size, embed_dim]
#   W_z: [embed_dim + hidden_dim, hidden_dim]  (update gate)
#   b_z: [hidden_dim]
#   W_r: [embed_dim + hidden_dim, hidden_dim]  (reset gate)
#   b_r: [hidden_dim]
#   W_h: [embed_dim + hidden_dim, hidden_dim]  (candidate)
#   b_h: [hidden_dim]
#   out_weight: [hidden_dim, vocab_size]
#   out_bias: [vocab_size]

MAGIC = b'LXNN'
VERSION = 1

# === Training corpus (built-in) ===
CORPUS = """The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.
How vexingly quick daft zebras jump! The five boxing wizards jump quickly.
In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the
ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit
down on or to eat: it was a hobbit-hole, and that means comfort.
It is a truth universally acknowledged, that a single man in possession of a good fortune,
must be in want of a wife. However little known the feelings or views of such a man may be
on his first entering a neighbourhood, this truth is so well fixed in the minds of the
surrounding families, that he is considered the rightful property of some one or other of
their daughters.
function fibonacci(n: number): number {
  if (n <= 1) return n;
  let a = 0, b = 1;
  for (let i = 2; i <= n; i++) {
    const temp = a + b;
    a = b;
    b = temp;
  }
  return b;
}

class TreeNode {
  val: number;
  left: TreeNode | null;
  right: TreeNode | null;
  constructor(val: number) {
    this.val = val;
    this.left = null;
    this.right = null;
  }
}

{"name": "LexiFold", "version": "2.0", "type": "compression", "backend": "neural"}
{"id": 1, "text": "Hello World", "score": 0.95, "tags": ["test", "example"]}
{"id": 2, "text": "Neural compression", "score": 0.87, "tags": ["neural", "compression"]}

2026-03-21 08:00:01.234 [INFO] server: Request processed successfully in 42ms
2026-03-21 08:00:02.567 [DEBUG] cache: Cache hit for key user:1234
2026-03-21 08:00:03.891 [WARN] auth: Token expires in 5 minutes
2026-03-21 08:00:04.123 [ERROR] db: Connection timeout after 30000ms

The rain in Spain falls mainly on the plain. She sells sea shells by the sea shore.
Peter Piper picked a peck of pickled peppers. A peck of pickled peppers Peter Piper picked.
Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.
The cat sat on the mat. The dog lay on the rug. The bird sat on the branch.
To be or not to be, that is the question. Whether it is nobler in the mind to suffer
the slings and arrows of outrageous fortune, or to take arms against a sea of troubles.
All that glitters is not gold. A stitch in time saves nine. Actions speak louder than words.
"""

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

def tanh(x):
    return np.tanh(np.clip(x, -15, 15))

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

class GRUModel:
    def __init__(self):
        scale = 0.1
        self.embed = np.random.randn(VOCAB_SIZE, EMBED_DIM).astype(np.float32) * scale
        input_size = EMBED_DIM + HIDDEN_DIM
        self.W_z = np.random.randn(input_size, HIDDEN_DIM).astype(np.float32) * scale
        self.b_z = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.W_r = np.random.randn(input_size, HIDDEN_DIM).astype(np.float32) * scale
        self.b_r = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.W_h = np.random.randn(input_size, HIDDEN_DIM).astype(np.float32) * scale
        self.b_h = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.out_w = np.random.randn(HIDDEN_DIM, VOCAB_SIZE).astype(np.float32) * scale
        self.out_b = np.zeros(VOCAB_SIZE, dtype=np.float32)

        # Adam state
        self.params = [self.embed, self.W_z, self.b_z, self.W_r, self.b_r,
                       self.W_h, self.b_h, self.out_w, self.out_b]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward_step(self, x_idx, h):
        """Single GRU step. Returns logits and new hidden state."""
        x = self.embed[x_idx]  # (embed_dim,)
        xh = np.concatenate([x, h])  # (embed_dim + hidden_dim,)

        z = sigmoid(xh @ self.W_z + self.b_z)
        r = sigmoid(xh @ self.W_r + self.b_r)

        xrh = np.concatenate([x, r * h])
        h_cand = tanh(xrh @ self.W_h + self.b_h)

        h_new = (1 - z) * h + z * h_cand
        logits = h_new @ self.out_w + self.out_b
        return logits, h_new

    def forward_sequence(self, seq):
        """Forward pass on a sequence. Returns list of logits."""
        h = np.zeros(HIDDEN_DIM, dtype=np.float32)
        all_logits = []
        for t in range(len(seq)):
            logits, h = self.forward_step(seq[t], h)
            all_logits.append(logits)
        return all_logits

    def train_step(self, seq):
        """Train on one sequence using BPTT with numerical gradients approximation.
        Uses a simplified approach: forward loss + finite differences on output layer,
        plus cross-entropy gradient for the full model via TBPTT."""
        h_states = [np.zeros(HIDDEN_DIM, dtype=np.float32)]
        x_embeds = []
        z_gates = []
        r_gates = []
        h_cands = []

        # Forward
        total_loss = 0.0
        for t in range(len(seq) - 1):
            x = self.embed[seq[t]].copy()
            h = h_states[-1]
            xh = np.concatenate([x, h])

            z = sigmoid(xh @ self.W_z + self.b_z)
            r = sigmoid(xh @ self.W_r + self.b_r)
            xrh = np.concatenate([x, r * h])
            h_cand = tanh(xrh @ self.W_h + self.b_h)
            h_new = (1 - z) * h + z * h_cand

            logits = h_new @ self.out_w + self.out_b
            probs = softmax(logits)

            target = seq[t + 1]
            total_loss -= np.log(probs[target] + 1e-10)

            x_embeds.append(x)
            z_gates.append(z)
            r_gates.append(r)
            h_cands.append(h_cand)
            h_states.append(h_new)

        T = len(seq) - 1
        if T == 0:
            return total_loss

        # Backward
        grads = {name: np.zeros_like(p) for name, p in zip(
            ['embed', 'W_z', 'b_z', 'W_r', 'b_r', 'W_h', 'b_h', 'out_w', 'out_b'],
            self.params)}

        dh_next = np.zeros(HIDDEN_DIM, dtype=np.float32)

        for t in reversed(range(T)):
            h_prev = h_states[t]
            h_cur = h_states[t + 1]
            x = x_embeds[t]
            z = z_gates[t]
            r = r_gates[t]
            h_cand = h_cands[t]

            # Output gradient
            logits = h_cur @ self.out_w + self.out_b
            probs = softmax(logits)
            target = seq[t + 1]
            dlogits = probs.copy()
            dlogits[target] -= 1.0  # (vocab,)

            grads['out_w'] += np.outer(h_cur, dlogits)
            grads['out_b'] += dlogits

            # dh from output + from next timestep
            dh = dlogits @ self.out_w.T + dh_next

            # GRU backward
            dh_cand = dh * z * (1 - h_cand ** 2)
            dz = dh * (h_cand - h_prev) * z * (1 - z)

            xh = np.concatenate([x, h_prev])
            xrh = np.concatenate([x, r * h_prev])

            grads['W_h'] += np.outer(xrh, dh_cand)
            grads['b_h'] += dh_cand

            grads['W_z'] += np.outer(xh, dz)
            grads['b_z'] += dz

            # dr from h_cand path
            d_rh = dh_cand @ self.W_h[:EMBED_DIM + HIDDEN_DIM].T
            d_rh_h = d_rh[EMBED_DIM:]  # gradient w.r.t. r*h
            dr = d_rh_h * h_prev * r * (1 - r)

            grads['W_r'] += np.outer(xh, dr)
            grads['b_r'] += dr

            # dh_prev from multiple paths
            dh_next = dh * (1 - z)
            dh_next += (dz @ self.W_z[EMBED_DIM:].T)
            dh_next += (dr @ self.W_r[EMBED_DIM:].T)
            dh_next += d_rh_h * r

            # dx for embedding gradient
            dx = np.zeros(EMBED_DIM, dtype=np.float32)
            dx += dz @ self.W_z[:EMBED_DIM].T
            dx += dr @ self.W_r[:EMBED_DIM].T
            dx += d_rh[:EMBED_DIM]
            grads['embed'][seq[t]] += dx

        # Clip gradients
        for name in grads:
            np.clip(grads[name], -CLIP_GRAD, CLIP_GRAD, out=grads[name])

        # Adam update
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        param_names = ['embed', 'W_z', 'b_z', 'W_r', 'b_r', 'W_h', 'b_h', 'out_w', 'out_b']
        for i, name in enumerate(param_names):
            g = grads[name]
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * g * g
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            self.params[i] -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + eps)

        # Sync references
        (self.embed, self.W_z, self.b_z, self.W_r, self.b_r,
         self.W_h, self.b_h, self.out_w, self.out_b) = self.params

        return total_loss / T

    def save_weights(self, path):
        with open(path, 'wb') as f:
            f.write(MAGIC)
            f.write(struct.pack('<IIII', VERSION, EMBED_DIM, HIDDEN_DIM, VOCAB_SIZE))
            for p in self.params:
                f.write(p.astype(np.float32).tobytes())
        size = os.path.getsize(path)
        print(f"Saved weights to {path} ({size} bytes, {sum(p.size for p in self.params)} params)")


def prepare_data(text):
    data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
    sequences = []
    for i in range(0, len(data) - SEQ_LEN, SEQ_LEN // 2):
        sequences.append(data[i:i + SEQ_LEN])
    return sequences


def main():
    print("=== LexiFold GRU Training ===\n")
    print(f"Model: Embedding({VOCAB_SIZE}→{EMBED_DIM}) → GRU({EMBED_DIM}→{HIDDEN_DIM}) → Linear({HIDDEN_DIM}→{VOCAB_SIZE})")

    model = GRUModel()
    total_params = sum(p.size for p in model.params)
    print(f"Parameters: {total_params:,} ({total_params * 4 / 1024:.1f} KB)\n")

    # Prepare data
    corpus = CORPUS * 3  # repeat for more training data
    sequences = prepare_data(corpus)
    print(f"Training sequences: {len(sequences)} (seq_len={SEQ_LEN})")

    # Train
    for epoch in range(EPOCHS):
        np.random.shuffle(sequences)
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for seq in sequences:
            loss = model.train_step(seq)
            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / n_batches
        bpc = avg_loss / np.log(2)  # bits per character
        elapsed = time.time() - t0
        print(f"Epoch {epoch + 1:3d}/{EPOCHS}  loss={avg_loss:.4f}  bpc={bpc:.3f}  time={elapsed:.1f}s")

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'entry', 'src', 'main', 'resources', 'rawfile')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'gru_weights.bin')
    model.save_weights(output_path)

    # Quick eval
    print("\n=== Quick Evaluation ===")
    test_text = "The quick brown fox jumps over the lazy dog."
    test_data = np.frombuffer(test_text.encode('utf-8'), dtype=np.uint8)
    h = np.zeros(HIDDEN_DIM, dtype=np.float32)
    total_bits = 0.0
    for i in range(len(test_data) - 1):
        logits, h = model.forward_step(test_data[i], h)
        probs = softmax(logits)
        total_bits -= np.log2(probs[test_data[i + 1]] + 1e-10)
    bpc = total_bits / (len(test_data) - 1)
    print(f"Test: '{test_text[:40]}...'  bpc={bpc:.3f}  (~{bpc/8*100:.1f}% compression ratio)")
    print("\nDone!")


if __name__ == '__main__':
    main()
