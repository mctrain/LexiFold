# LexiFold

面向 HarmonyOS 的神经网络启发式文本压缩引擎，受 [ts_zip](https://bellard.org/ts_zip/) 和 [NNCP](https://bellard.org/nncp/) 工程思路启发。

## 核心思想

LexiFold 的压缩原理与传统压缩器（gzip/zstd 的 LZ77 字典匹配）完全不同，采用 **预测建模 + 算术编码** 路线：

```
预测模型估计「下一个字节的概率分布」
        ↓
算术编码器利用这个分布，把高概率的字节编得更短
        ↓
概率越准 → 熵越低 → 压缩率越好
```

如果模型预测"下一个字节是 `e`"的概率是 90%，编码 `e` 只需约 0.15 bit（而非 8 bit）。模型预测越准，压缩率越高。这为未来接入神经网络预测器（NPU 推理）提供了天然的架构优势——只需替换概率来源，整个编码管线不变。

---

## 数据流

### 压缩

```
"Hello World"
    │
    ▼
TextPreprocessor（v0: 恒等变换）
    │
    ▼
ByteTokenizer（UTF-8 编码 → [72, 101, 108, 108, 111, 32, 87, ...]）
    │
    ▼
ChunkProcessor（按 64KB 分块，每块独立压缩）
    │
    │   ┌──────────────────────────────────────────┐
    │   │  对块中的每个字节 symbol:                  │
    │   │    1. backend.getFrequency(symbol)        │
    │   │       → 获取当前概率分布                   │
    │   │    2. encoder.encode(cumLow, cumHigh)      │
    │   │       → 算术编码                          │
    │   │    3. backend.update(symbol)               │
    │   │       → 更新模型状态                       │
    │   └──────────────────────────────────────────┘
    │
    ▼
ArchiveFormat → .lxf 文件（文件头 + 块头 + 压缩数据 + CRC32）
```

### 解压（严格对称）

```
.lxf 文件
    │
    ▼
ArchiveFormat（解析文件头，校验 magic / version / CRC32）
    │
    ▼
ChunkProcessor（逐块解压）
    │
    │   ┌──────────────────────────────────────────┐
    │   │  对每个位置 i = 0..originalSize-1:        │
    │   │    1. backend.getFrequency(...)           │
    │   │       → 获取当前概率分布（与压缩端一致）    │
    │   │    2. decoder.getScaledValue(total)        │
    │   │       → 在分布中定位                       │
    │   │    3. backend.findSymbol(value)            │
    │   │       → 恢复原始字节                       │
    │   │    4. backend.update(symbol)               │
    │   │       → 更新模型状态（与压缩端一致）        │
    │   └──────────────────────────────────────────┘
    │
    ▼
CRC32 校验 → ByteTokenizer.detokenize → 原始文本
```

**无损保证的关键**：压缩端和解压端执行完全相同的 `getFrequency()` → `update()` 序列。两端看到相同的上下文，产生相同的概率分布，任何数值分歧都会导致解压失败。

---

## 架构分层

```
┌─────────────────────────────────────────────┐
│  UI 层 (Index.ets)                           │
│  Tabs 导航 · 卡片布局 · 结构化指标展示         │
├─────────────────────────────────────────────┤
│  服务层 (CompressionService.ets)              │
│  任务编排 · 文件 I/O · 错误映射               │
├─────────────────────────────────────────────┤
│  压缩核心层                                   │
│  ┌───────────┐ ┌──────────┐ ┌────────────┐  │
│  │ Tokenizer │ │Predictor │ │ Arithmetic │  │
│  │ byte-256  │ │ Backend  │ │   Coder    │  │
│  └───────────┘ └──────────┘ └────────────┘  │
│  ┌───────────┐ ┌──────────┐ ┌────────────┐  │
│  │  Archive  │ │  Chunk   │ │  Checksum  │  │
│  │  Format   │ │Processor │ │   CRC32    │  │
│  └───────────┘ └──────────┘ └────────────┘  │
├─────────────────────────────────────────────┤
│  后端层 (可替换)                              │
│  CpuReference · MockStep · NPU · Adaptive   │
└─────────────────────────────────────────────┘
```

### 关键文件

| 文件 | 职责 |
|------|------|
| `core/Types.ets` | 全部类型、常量、接口定义、Profile 配置 |
| `core/ArithmeticCoder.ets` | 24-bit 算术编码器 + 解码器 (Witten-Neal-Cleary) |
| `core/PredictorBackend.ets` | 后端接口 + CpuReference (order-0 自适应) + Mock (均匀分布) + 工厂 |
| `core/ChunkProcessor.ets` | 分块压缩/解压编排，CRC32 校验 |
| `core/ArchiveFormat.ets` | `.lxf` 容器格式二进制读写 |
| `core/Checksum.ets` | CRC32 实现 (0xEDB88320 多项式) |
| `core/Tokenizer.ets` | ByteTokenizer: UTF-8 ↔ byte 数组 |
| `core/TextPreprocessor.ets` | 文本预处理 (v0 恒等，v1 可逆变换) |
| `service/CompressionService.ets` | 高层服务：压缩/解压/验证 |
| `pages/Index.ets` | HarmonyOS UI：Tabs、卡片、指标 |

---

## 各层技术细节

### 算术编码器

采用经典 **Witten-Neal-Cleary** 方案，24-bit 精度：

```
初始状态: low = 0, high = 2^24 - 1 (= 16,777,215)

编码一个符号 (cumLow, cumHigh, total):
  range = high - low + 1
  high  = low + floor(range × cumHigh / total) - 1
  low   = low + floor(range × cumLow / total)

  归一化: 当 low/high 都落在同一半区间时，输出确定的 bit
    - 都在下半 → 输出 0
    - 都在上半 → 输出 1
    - 跨越中点但靠近 → 累积 pending bit（bit-plus-follow 技术）
```

**为什么 24-bit**：ArkTS 的 number 是 64-bit double，可精确表示 2^53 以内整数。24-bit range 最小约 4M，与频率总和 16K 相除分辨率 > 256，足够精确且避免了 32-bit 有符号整数的位运算陷阱。

### 预测后端 (v0: Order-0 自适应频率模型)

```
初始: counts[0..255] = [1, 1, 1, ..., 1]  (Laplace 平滑)
      total = 256

编码/解码每个字节后:
  counts[byte]++
  total++
  if total >= 16384:    // 防止算术编码器精度不足
    所有 count 减半 (floor, 最小保持 1)
```

本质是统计"到目前为止每个字节值出现了多少次"，用历史频率作为预测概率。这不是演示用的假模型——它真正跟踪字节频率，对英文文本可达 ~62% 压缩率。

**局限性**：没有上下文感知。看到 `t-h-` 后不知道下一个大概率是 `e`，只知道全局 `e` 出现频率最高。只能利用字符频率不均匀这一个特征。

**接口可替换**：未来接入 RWKV/Transformer 时，只需实现同一个 `PredictorBackend` 接口：

```
当前:  counts[byte] / total        → 概率分布 → 算术编码器
未来:  model.step(token, state)     → logits → softmax → 概率分布 → 算术编码器
       ▲                             ▲
       RWKV / 小 Transformer         NPU 推理
```

### 分块处理

- 默认 64KB/块，每块独立 `reset()` 模型状态
- 每块存储：压缩数据长度 + 原始数据长度 + 原始数据 CRC32
- 解压逐块校验，任何 CRC32 不匹配立即报错

### 归档格式 (.lxf)

```
┌───────────────────────── 26 字节文件头 ──────────────────────────┐
│ "LXFD"(4B) │ ver(1B) │ flags(1B) │ tokID(1B) │ prepID(1B)      │
│ backendID(2B LE) │ modelID(2B LE) │ chunkSize(4B LE)           │
│ originalSize(4B LE) │ numChunks(2B LE) │ headerCRC32(4B LE)    │
└─────────────────────────────────────────────────────────────────┘
┌──────────── 块 0 ─────────────┐ ┌──────────── 块 1 ────────────┐
│ compSize(4B) origSize(4B)     │ │ compSize(4B) origSize(4B)    │ ...
│ crc32(4B)    compressedData   │ │ crc32(4B)    compressedData  │
└───────────────────────────────┘ └──────────────────────────────┘
```

元信息完整记录 tokenizer / backend / model ID，确保解压端使用完全一致的配置。

---

## 压缩效果实测

| 测试数据 | 原始大小 | 压缩后 | 比率 |
|---------|---------|--------|------|
| 重复英文文本 (5.6KB) | 5,390 B | 3,356 B | **62.3%** |
| 英文段落 | 123 B | 105 B | 85.4% |
| JSON 数据 | 56 B | 51 B | 91.1% |
| 重复 pattern "abc..." | 30 B | 23 B | 76.7% |
| UTF-8 中文 | 42 B | 41 B | 97.6% |
| 伪随机数据 (1KB) | 1,000 B | 1,023 B | 102.3% |
| 全 256 字节值 | 256 B | 274 B | 107.0% |

Order-0 模型对重复模式和自然语言有效，对随机/均匀数据略微膨胀（符合信息论预期）。换成神经网络后端后，英文压缩率预计可从 62% 降到 15-25%。

---

## 设计决策

| 决策 | 原因 |
|------|------|
| ts_zip 路线而非 LZ77 | 为接入神经网络预测器铺路，LZ77 无法利用模型能力 |
| 固定模型推理 + 算术编码 | 成熟可靠，encode/decode 天然对称 |
| Order-0 作为 v0 后端 | 最简单的真实概率模型，验证完整链路正确性 |
| 24-bit 精度 | 避免 JS 整数溢出，精度足够 |
| Byte-level tokenizer | 无损、通用、零依赖，不需要词表文件 |
| 分块 + CRC32 | 内存可控 + 完整性校验 + 损坏隔离 |
| 接口抽象 (PredictorBackend) | 后续换模型不动编码层 |
| 不做 NNCP 式在线训练 | v0 追求可运行闭环，训练是 v2+ 的事 |

---

## 构建与运行

需要 **DevEco Studio** + HarmonyOS SDK 6.0.2(22)。

```bash
# DevEco Studio
1. 打开项目
2. Build → Build Hap(s)/APP(s) → Build Hap(s)
3. Run → Run 'entry'

# 或使用 hvigorw CLI
hvigorw assembleHap
hdc install entry/build/default/outputs/default/entry-default-unsigned.hap
hdc shell aa start -a EntryAbility -b com.ohos.lexifold
```

### 离线验证核心算法（无需设备）

```bash
npx tsx tools/verify_roundtrip.ts
```

输出 9 项 round-trip 测试，全部 PASS 表示算术编码器和预测模型工作正常。

---

## 当前限制 (v0)

- 仅支持文本输入（无通用二进制压缩）
- Order-0 自适应模型——无上下文感知，压缩率有限
- 同步压缩（大文件可能阻塞 UI）
- 无 NPU 加速

---

## 演进路线

```
v0 (当前) ──→ v1 ────────────→ v2 ─────────────→ 实验
byte-256      + 可逆文本预处理    + NPU 后端         + NNCP 式
order-0       + tokenizer 切换   + 小模型推理          在线自适应
              + profile 管理     + step(tok, state)
                                   → logits
```

核心架构不变——**只替换 `PredictorBackend` 的实现**。编码器、归档格式、校验逻辑、UI 层全部复用。

详见 `docs/implementation_plan.md`、`docs/architecture.md`、`docs/npu_integration.md`。
