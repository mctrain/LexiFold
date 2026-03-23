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

### 预测后端 (v2: PPM Order-3 Blended, C++ Native)

PPM (Prediction by Partial Matching) 是经典的统计上下文模型。LexiFold v2 实现了 Order-3 变体，通过 N-API 桥接 C++ 引擎。

**核心思想**：同时维护 4 层上下文，融合预测：

```
输入序列: ...t-h-e-_-f-o-?    要预测 ? 位置的字节

Order-3: 看前 3 字节 "_fo" → "在 _fo 之后通常出现什么？" → 几乎确定是 x
Order-2: 看前 2 字节 "fo"  → "在 fo 之后通常出现什么？"  → 较强预测
Order-1: 看前 1 字节 "o"   → "在 o 之后通常出现什么？"   → 一般预测
Order-0: 无上下文           → "英文中哪个字节最常见？"     → 最弱预测

融合: blended[byte] = (w3*p3 + w2*p2 + w1*p1 + w0*p0) / (w3+w2+w1+w0)
```

**动态权重**：高阶上下文见过足够多数据时权重更大，新出现的上下文退化为低阶预测：

```
w0 = 1 (恒定基线)
w1 = min(order1_seen / 4, 16)     新上下文 → 低权重，成熟上下文 → 16x
w2 = min(order2_seen / 2, 32)     最多 32x
w3 = min(order3_seen, 64)         最多 64x
```

**存储**：Order-0/1 用固定数组（130KB），Order-2/3 用哈希表惰性分配（只在上下文首次出现时创建）。

**各版本预测能力对比**：

| 序列 `...t-h-e-_-f-o-?` | v0 Order-0 | v1 Order-1 | v2 PPM-3 |
|--------------------------|-----------|-----------|----------|
| 能看到的上下文 | 无 | `o` | `_fo` + `fo` + `o` + 全局 |
| 知道 `th` 常见？ | 不知道 | 知道 | 知道 |
| 知道 `the` 常见？ | 不知道 | 不知道 | **知道** |
| 知道 `fox` 中 `fo→x`？ | 不知道 | 不知道 | **知道** |
| 预测精度 | ~5.0 bits/byte | ~3.4 bits/byte | ~0.9 bits/byte |

**为什么 PPM-3 不够、还需要神经网络？**

PPM 的上下文窗口固定（最多 3 字节）。而自然语言的依赖关系可跨越段落甚至全文（主题、人名、变量名复现）。神经网络通过隐藏状态可记住任意长上下文，v2 建立的 N-API 管线已为此做好准备——只需替换 `PredictorBackend` 实现。

```
v0:  counts[byte] / total              → 概率分布 → 算术编码器
v1:  counts[prev][byte] / total[prev]  → 概率分布 → 算术编码器
v2:  blend(o0, o1, o2, o3)             → 概率分布 → 算术编码器  ← C++ N-API
未来: model.step(token, state) → logits → 概率分布 → 算术编码器  ← NPU 推理
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

## 压缩效果对比

10 种测试数据 × 4 个后端 + gzip -9 基准。**数值越低越好**（压缩后 / 原始大小）。

| 测试数据 | 大小 | v0 Order-0 | v1 Order-1 | v2 PPM-3 | v2 GRU | gzip -9 |
|---------|-----:|:---:|:---:|:---:|:---:|:---:|
| English prose (5.4KB) | 5,390 | 62.3% | 42.6% | **11.1%** | 27.6% | 6.4% |
| Short English (122B) | 122 | 86.1% | 95.9% | 91.0% | **4.1%** | 113.1% |
| JSON lines (3.0KB) | 2,649 | 59.7% | 53.4% | **25.3%** | 102.5% | 18.4% |
| Chinese text (8.7KB) | 4,080 | 72.4% | 60.6% | **20.0%** | 163.0% | 5.9% |
| TypeScript code (3.8KB) | 4,060 | 58.7% | 52.0% | **17.1%** | 65.8% | 7.4% |
| HTML markup (2.2KB) | 3,173 | 61.6% | 51.5% | **21.2%** | 127.5% | 13.4% |
| CSV data (5.6KB) | 3,957 | 65.9% | 63.8% | **38.9%** | 134.3% | 31.7% |
| Log file (5.5KB) | 6,565 | 66.6% | 58.0% | **26.2%** | 42.1% | 15.6% |
| Multilingual (2.8KB) | 2,746 | 74.5% | 60.8% | **23.1%** | 133.8% | 11.6% |
| Random bytes (4KB) | 4,096 | 101.1% | 100.1% | **99.8%** | 159.4% | 101.0% |

### 各版本说明

| 版本 | 后端 | 实现 | 说明 |
|------|------|------|------|
| **v0** | Order-0 Adaptive | ArkTS | 全局字节频率统计，无上下文 |
| **v1** | Order-1 Adaptive | ArkTS | 按前一字节分 256 个上下文 |
| **v2 PPM** | PPM Order-3 Blended | C++ (N-API) | 融合 4 层上下文，自适应，无需训练 |
| **v2 GRU** | GRU Neural | C++ (N-API) | 36K 参数 char-level GRU，训练在 ~7KB 内置语料上 |
| — | gzip -9 | 参考 | 传统压缩器（LZ77 + Huffman），最高级别 |

### 版本间演进分析

**v0 → v1（+32% 提升）：**
- 加入 1 字节上下文，英文 62.3% → 42.6%。
- 但短文本 Order-1 反而退化（上下文来不及建立统计）。

**v1 → v2（+73% 提升，架构跃迁）：**
- PPM Order-3 融合多层上下文，英文 42.6% → **11.1%**（提升 74%），接近 gzip 的 6.4%。
- JSON 53.4% → 25.3%，源码 52.0% → 17.1%，日志 58.0% → 26.2%。
- **首次在多数数据类型上接近甚至挑战 gzip**：HTML 21.2% vs gzip 13.4%，CSV 38.9% vs 31.7%。
- v2 引入 C++ native 层（N-API 桥接），为未来 ONNX / NPU 模型集成建立了完整管线。

**v2 GRU Neural（过拟合分析）：**
- GRU 在训练语料中出现过的文本上表现极好：Short English **4.1%**（原句出现在训练集中），远超所有方法包括 gzip。
- 在训练分布附近的文本上有限泛化：English prose 27.6%，Log file 42.1%。
- 在训练集中未覆盖的数据类型上**严重膨胀**：JSON 102.5%、中文 163.0%、HTML 127.5%、CSV 134.3%。
- **根因**：36K 参数 tiny GRU 在 ~7KB 内置语料上训练 30 epoch，记忆力 > 泛化力。
- **启示**：神经网络后端的潜力已被验证（训练分布内可达 4%），但需要**大规模多样化语料**训练才能在泛化场景超越 PPM。

**v2 PPM vs gzip：**
- gzip 仍在大部分场景领先，因为 LZ77 能匹配长距离重复模式（PPM 靠短上下文预测，对长重复不敏感）。
- 但 LexiFold v2 在 **短文本** 上始终优于 gzip（PPM 91.0%、GRU 4.1% vs gzip 113.1%）——预测式压缩无头部开销。
- **关键优势**：LexiFold 的后端可随时替换为更强的神经网络，而 gzip 架构已固化。

### 运行基准测试

```bash
npx tsx tools/benchmark.ts
```

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
