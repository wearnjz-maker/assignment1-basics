# CS336 作业 1：基础（中文精译）

> 原文：`cs336_assignment1_basics.pdf`，版本 1.0.6，CS336 教学团队，2025 年春季。
>
> 本文保留全部主要任务、接口、公式、测试命令、实验要求与资源限制；背景性说明作了适度压缩。它是题目翻译，不包含题目答案或实现代码。术语首次出现时保留英文，便于与代码和论文对应。

## 1. 作业概览

本作业要求你从零构建训练标准 Transformer 语言模型（LM）所需的全部组件，并实际训练若干模型。

你将实现：

1. 字节对编码分词器（byte-pair encoding tokenizer，BPE）；
2. Transformer 语言模型；
3. 交叉熵损失与 AdamW 优化器；
4. 支持保存、加载模型及优化器状态的训练循环。

你将运行：

1. 在 TinyStories 上训练 BPE 分词器；
2. 用训练好的分词器把数据集转换为整数 token ID 序列；
3. 在 TinyStories 上训练 Transformer LM；
4. 用模型生成样本并评估困惑度（perplexity）；
5. 在 OpenWebText（OWT）上训练模型，并把结果提交到排行榜。

### 可以使用什么

你需要从零构建核心组件。除以下内容外，不得使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的现成定义：

- `torch.nn.Parameter`；
- `torch.nn` 的容器类，例如 `Module`、`ModuleList`、`Sequential`；
- `torch.optim.Optimizer` 基类。

其他 PyTorch 功能可以使用。判断标准是：使用某个现成功能是否破坏了本作业“从零实现”的目的。

课程允许仅针对低层编程问题或语言模型的高层概念问题询问 LLM，但禁止让 LLM 直接解决题目。建议关闭 AI 自动补全。

### 代码结构与提交

- `cs336_basics/*`：编写自己实现的地方；起始时基本没有脚手架。
- `tests/adapters.py`：把你实现的功能接到测试上。这里应当只有胶水逻辑，不应放实质实现。
- `tests/test_*.py`：必须通过的测试，不要修改。

提交内容：

- `writeup.pdf`：所有书面题目的排版答案；
- `code.zip`：你编写的全部代码。

数据集为 TinyStories 与 OpenWebText，均是大型纯文本文件。下载方式见仓库 `README.md`。

低资源说明：教学团队的实现可以在 36 GB 内存的 Apple M3 Max 上，用 MPS 在 5 分钟内、CPU 在约 30 分钟内训练出能够生成较流畅儿童故事的小模型。后续章节给出了 CPU/MPS 的缩小配置。

## 2. 字节对编码（BPE）分词器

本部分实现字节级 BPE。任意 Unicode 字符串先表示为字节序列；训练后，分词器把字符串编码为整数 token 序列，也能把整数序列解码回文本。

### 2.1 Unicode 标准

Unicode 把字符映射到整数码点（code point）。Python 的 `ord()` 把单个 Unicode 字符转换为整数，`chr()` 则执行反向转换。

#### 题目 `unicode1`：理解 Unicode（1 分）

1. `chr(0)` 返回什么 Unicode 字符？提交一句话。
2. 该字符的字符串表示 `__repr__()` 与打印表示有何不同？提交一句话。
3. 该字符出现在文本中时会怎样？可以在解释器中比较 `chr(0)`、`print(chr(0))` 以及把它拼入普通字符串后的结果。提交一句话。

### 2.2 Unicode 编码

直接在约 15 万个 Unicode 码点上训练分词器会得到巨大且稀疏的词表，因此要先使用 Unicode 编码把字符转换为字节。Unicode 标准定义了 UTF-8、UTF-16 和 UTF-32；互联网主要使用 UTF-8。

Python 中，字符串的 `encode("utf-8")` 返回 `bytes`；遍历 `bytes` 可得到 0 到 255 的整数；`decode("utf-8")` 可还原字符串。一个 Unicode 字符不一定对应一个字节。字节级分词的初始词表只需 256 项，并且任何输入文本都不会产生词表外 token。

#### 题目 `unicode2`：Unicode 编码（3 分）

1. 与 UTF-16 或 UTF-32 相比，为什么可能更愿意在 UTF-8 字节上训练分词器？可比较多种输入字符串的编码结果。提交 1–2 句话。
2. 题目给出的错误函数逐字节执行 UTF-8 解码。给出一个会产生错误结果的输入字节串，并用一句话解释错误原因。
3. 给出一个无法解码为任何 Unicode 字符的两字节序列，并用一句话解释。

### 2.3 子词分词

纯字节分词解决了词表外问题，但序列会很长，训练计算量和长期依赖难度都会增加。子词分词位于词级和字节级之间：它用更大的词表换取更短的序列。例如，经常出现的 `b'the'` 可以成为一个词表项，把 3 个字节 token 压缩为 1 个 token。

BPE 反复找出最频繁的相邻 token 对，把它们合并为新的 token。足够常见的单词或片段最终会成为一个词表项。

### 2.4 训练 BPE 分词器

BPE 训练包括三个主要步骤：

1. **初始化词表**：字节级 BPE 的初始词表包含全部 256 个可能的字节值。
2. **预分词（pre-tokenization）**：先用粗粒度规则切分语料，在每个 pre-token 内统计相邻字节，不跨 pre-token 边界合并。本作业采用 GPT-2 风格的正则表达式：

   `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`

   实际实现时建议使用 `re.finditer`，避免把全部 pre-token 一次性存入列表。
3. **计算合并（merges）**：反复统计相邻 token 对，选择频率最高的一对 `(A, B)`，把每次出现替换为新 token `AB`，并把新 token 加入词表。频率并列时，选择字典序更大的 token 对。不得跨 pre-token 边界统计或合并。

特殊 token（例如 `<|endoftext|>`）必须始终作为单个 token 保存，并加入词表以获得固定 ID。预分词前应把特殊 token 当作边界切开，防止其两侧发生合并。

效率提示：预分词通常是瓶颈，可以用多进程并行，并把语料分块边界放在特殊 token 起点。朴素合并每轮重新扫描全部 pair 会很慢；只有与本轮合并重叠的 pair 计数会发生变化，因此可缓存 pair 计数并增量更新。先在验证集等较小的“调试数据集”上验证行为，并用 `cProfile` 或 `scalene` 定位真正瓶颈。

#### 题目 `train_bpe`：训练 BPE 分词器（15 分）

实现一个函数，根据输入文本文件训练字节级 BPE，至少接受：

- `input_path: str`：训练文本路径；
- `vocab_size: int`：最终词表最大大小，包含初始字节、合并产生的词表项和特殊 token；
- `special_tokens: list[str]`：加入词表的特殊 token；它们不应以其他方式影响 BPE 训练。

返回：

- `vocab: dict[int, bytes]`：token ID 到 token 字节串的映射；
- `merges: list[tuple[bytes, bytes]]`：按创建顺序排列的合并列表。

先完成 `adapters.run_train_bpe`，再运行：`uv run pytest tests/test_train_bpe.py`。实现应通过全部测试。

可选：可用 C++ 或 Rust 实现关键性能部分，但需要处理好 Python 内存读取与复制，并留下构建说明或仅依赖 `pyproject.toml` 构建。

#### 题目 `train_bpe_tinystories`：在 TinyStories 上训练 BPE（2 分）

1. 在 TinyStories 上训练最大词表大小为 10,000 的字节级 BPE，并加入 `<|endoftext|>`。把词表和 merges 序列化到磁盘。报告耗时、内存、词表中最长 token，以及它是否合理。资源限制：不使用 GPU，最多 30 分钟、30 GB RAM。提交 1–2 句话。
2. 对代码做性能分析：训练流程的哪一部分耗时最多？提交 1–2 句话。

#### 题目 `train_bpe_expts_owt`：在 OpenWebText 上训练 BPE（2 分）

1. 在 OWT 上训练最大词表大小为 32,000 的字节级 BPE，序列化词表和 merges。最长 token 是什么，是否合理？资源限制：不使用 GPU，最多 12 小时、100 GB RAM。提交 1–2 句话。
2. 比较在 TinyStories 与 OWT 上训练得到的分词器。提交 1–2 句话。

### 2.5 编码与解码

编码过程与训练相似：先预分词，把各 pre-token 表示为 UTF-8 字节序列；然后在各 pre-token 内按训练时的创建顺序应用 merges，不跨边界合并。特殊 token 必须得到正确保留。

对于无法整体放入内存的大文件，需要分块流式编码，并保证 token 不跨块边界，否则结果会不同于一次性编码整个文本。

解码时，根据 ID 查词表得到字节串，依次连接后解码为 Unicode。如果 ID 序列组合出的字节不是合法 Unicode，使用正式替换字符 U+FFFD；`bytes.decode(errors='replace')` 可实现这种行为。

#### 题目 `tokenizer`：实现分词器（15 分）

实现 `Tokenizer` 类：给定词表与 merges，在文本和 token ID 之间双向转换，并支持用户指定的特殊 token；若特殊 token 尚未在词表中，应加入词表。建议接口：

- `__init__(vocab, merges, special_tokens=None)`；
- `from_files(vocab_filepath, merges_filepath, special_tokens=None)`；
- `encode(text: str) -> list[int]`；
- `encode_iterable(iterable: Iterable[str]) -> Iterator[int]`：惰性地产生 token ID，用于大文件；
- `decode(ids: list[int]) -> str`。

完成 `adapters.get_tokenizer`，运行：`uv run pytest tests/test_tokenizer.py`。

#### 题目 `tokenizer_experiments`：分词器实验（4 分）

1. 从 TinyStories 和 OWT 各采样 10 个文档，分别用之前训练的 10K 和 32K 词表分词器编码。计算各自压缩率（bytes/token）。提交 1–2 句话。
2. 用 TinyStories 分词器编码 OWT 样本会怎样？比较压缩率或作定性描述。提交 1–2 句话。
3. 估计分词器吞吐量（例如 bytes/s），推算处理 825 GB 的 Pile 数据集需要多久。提交 1–2 句话。
4. 用相应分词器把两个数据集的训练集和开发集编码为整数 token ID 序列。建议序列化为 `uint16` 的 NumPy 数组。解释为什么 `uint16` 合适。提交 1–2 句话。

## 3. Transformer 语言模型

语言模型输入形状为 `(batch_size, sequence_length)` 的 token ID，输出每个位置对下一个 token 的预测 logits，形状为 `(batch_size, sequence_length, vocab_size)`。训练时用这些预测计算交叉熵；生成时取最后一个位置的分布来选取下一个 token，再把它追加到输入并重复。

总体结构：token embedding → 多个 Transformer block → 最终归一化 → 输出线性层（LM head）。每个 block 接收并返回 `(batch_size, sequence_length, d_model)`，用自注意力聚合序列信息，并用前馈网络作非线性变换。

批维、序列位置和注意力头都可视为“批式维度”。建议把它们放在 tensor 形状前部，使用 PyTorch 广播、批量矩阵乘法或 `einsum`，避免 Python 循环。

### 3.1 参数初始化

使用题目指定的截断正态初始化：线性层权重按给定方差初始化并截断到 `[-3σ, 3σ]`；embedding 使用方差 1、截断到 `[-3, 3]`；RMSNorm 增益初始化为 1。使用 `torch.nn.init.trunc_normal_`。

#### 题目 `linear`：实现线性层（1 分）

实现继承 `torch.nn.Module` 的无偏置 `Linear`，完成 `y = Wx`，接口与 `nn.Linear` 类似但没有 bias：

- `__init__(in_features, out_features, device=None, dtype=None)`；
- `forward(x: torch.Tensor) -> torch.Tensor`。

权重以 `W` 而非 `Wᵀ` 的形式存入 `nn.Parameter`。不得使用 `nn.Linear` 或 `nn.functional.linear`。完成 `adapters.run_linear`，运行 `uv run pytest -k test_linear`。

#### 题目 `embedding`：实现 Embedding（1 分）

实现继承 `torch.nn.Module` 的 embedding lookup，不得使用 `nn.Embedding` 或 `nn.functional.embedding`。embedding 矩阵形状为 `(vocab_size, d_model)`，通过形状为 `(batch_size, sequence_length)` 的 `LongTensor` token ID 索引。

建议接口：

- `__init__(num_embeddings, embedding_dim, device=None, dtype=None)`；
- `forward(token_ids: torch.Tensor) -> torch.Tensor`。

完成 `adapters.run_embedding`，运行 `uv run pytest -k test_embedding`。

### 3.2 Pre-Norm Transformer Block

每个 block 有两个子层：多头自注意力与逐位置前馈网络。原始 Transformer 在各子层和残差相加后归一化，称为 post-norm。本作业使用 pre-norm：先归一化，再执行子层，最后加残差；最后一个 Transformer block 之后还要额外归一化。这样会形成较干净的残差流，通常有利于梯度传播与训练稳定性。

#### RMSNorm

对激活向量 `a ∈ R^{d_model}`：

`RMSNorm(a_i) = (a_i / RMS(a)) g_i`

其中 `RMS(a) = sqrt((1/d_model) Σ_i a_i² + ε)`，`g_i` 是可学习增益，通常 `ε = 1e-5`。为防止平方溢出，归一化计算前把输入上转为 `float32`，结果再转回原 dtype。

##### 题目 `rmsnorm`：实现 RMSNorm（1 分）

实现 `torch.nn.Module`，建议接口：

- `__init__(d_model: int, eps: float = 1e-5, device=None, dtype=None)`；
- `forward(x)`：输入输出形状均为 `(batch_size, sequence_length, d_model)`。

完成 `adapters.run_rmsnorm`，运行 `uv run pytest -k test_rmsnorm`。

#### SwiGLU 前馈网络

本作业使用现代 LLM 常见的 SiLU/Swish 激活和 GLU 门控，并省略线性层 bias：

`SiLU(x) = x · sigmoid(x)`

`FFN(x) = W2 (SiLU(W1 x) ⊙ W3 x)`

其中 `W1, W3 ∈ R^{d_ff × d_model}`，`W2 ∈ R^{d_model × d_ff}`。规范选择约为 `d_ff = (8/3)d_model`。

##### 题目 `positionwise_feedforward`：实现逐位置前馈网络（2 分）

实现由 SiLU 与 GLU 组成的 SwiGLU。可以使用 `torch.sigmoid` 以保证数值稳定。`d_ff` 取约 `(8/3)d_model`，并调整为 64 的倍数以更好利用硬件。完成 `adapters.run_swiglu`，运行 `uv run pytest -k test_swiglu`。

#### RoPE 相对位置编码

旋转位置编码（Rotary Position Embeddings，RoPE）把 query/key 向量的相邻维度两两视为二维向量，并按 token 位置旋转。该层没有可学习参数。无需构造完整旋转矩阵；正弦和余弦值可预计算，并通过 `register_buffer(persistent=False)` 保存和跨层复用。

##### 题目 `rope`：实现 RoPE（2 分）

实现 `RotaryPositionalEmbedding`：

- `__init__(theta: float, d_k: int, max_seq_len: int, device=None)`；
- `forward(x, token_positions)`：`x` 形状为 `(..., seq_len, d_k)`，输出同形；`token_positions` 形状为 `(..., seq_len)`。

必须支持任意数量的批式维度，并按 token position 从预计算的 sin/cos 中取值。完成 `adapters.run_rope`，运行 `uv run pytest -k test_rope`。

#### Softmax 与缩放点积注意力

`softmax(v)_i = exp(v_i) / Σ_j exp(v_j)`。

为避免大数指数溢出，可利用 softmax 对整体平移不变的性质，先沿目标维减去最大值。

##### 题目 `softmax`：实现 softmax（1 分）

函数接收 tensor 和维度 `i`，沿第 `i` 维应用 softmax，输出同形；必须使用减最大值技巧保证数值稳定。完成 `adapters.run_softmax`，运行 `uv run pytest -k test_softmax_matches_pytorch`。

缩放点积注意力为：

`Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k)) V`

布尔 mask 中 `True` 表示允许 query 关注对应 key，`False` 表示禁止。可在 softmax 前给禁止位置加 `-∞`。

##### 题目 `scaled_dot_product_attention`：实现缩放点积注意力（5 分）

支持 query/key 形状 `(batch_size, ..., seq_len, d_k)`，value 形状 `(batch_size, ..., seq_len, d_v)`，其中 `...` 是任意批式维度。还要支持可选的 `(seq_len, seq_len)` 布尔 mask：允许位置的注意力概率合计为 1，禁止位置概率为 0。

完成 `adapters.run_scaled_dot_product_attention`，分别运行三维和四维输入测试：

- `uv run pytest -k test_scaled_dot_product_attention`；
- `uv run pytest -k test_4d_scaled_dot_product_attention`。

#### 因果多头自注意力

多头注意力把 embedding 维切分为 `h` 个头，各头独立执行 attention，再拼接并经输出投影。自注意力中的 Q、K、V 都由同一输入 `x` 投影得到。取 `d_k = d_v = d_model / h`。

因果 mask 要保证位置 `i` 只能关注 `j ≤ i`，不能读取未来 token。RoPE 只应用于 query 和 key，不应用于 value；head 维应作为批式维度处理，各头使用相同位置旋转。

##### 题目 `multihead_self_attention`：实现因果多头自注意力（5 分）

实现一个 `torch.nn.Module`，至少接受 `d_model` 和 `num_heads`。完成 `adapters.run_multihead_self_attention`，运行 `uv run pytest -k test_multihead_self_attention`。

### 3.3 完整 Transformer LM

一个 pre-norm block 的第一子层为：

`y = x + MultiHeadSelfAttention(RMSNorm(x))`

第二子层同样先 RMSNorm，再经过 FFN，最后加入残差。

#### 题目 `transformer_block`：实现 Transformer block（3 分）

至少接受：

- `d_model`：输入隐藏维；
- `num_heads`：注意力头数；
- `d_ff`：前馈网络内部维度。

完成 `adapters.run_transformer_block`，运行 `uv run pytest -k test_transformer_block`。提交能够通过测试的 Transformer block。

#### 题目 `transformer_lm`：实现 Transformer LM（3 分）

把 token embedding、`num_layers` 个 block、最终归一化和输出 embedding/LM head 组合起来。除 block 参数外，至少接受：

- `vocab_size`；
- `context_length`；
- `num_layers`。

完成 `adapters.run_transformer_lm`，运行 `uv run pytest -k test_transformer_lm`。提交能够通过测试的 LM 模块。

#### 题目 `transformer_accounting`：Transformer 资源核算（5 分）

矩阵 `A ∈ R^{m×n}` 与 `B ∈ R^{n×p}` 相乘需要 `2mnp` FLOPs。

1. 对 GPT-2 XL 配置（词表 50,257，上下文 1,024，48 层，`d_model=1600`，25 头，`d_ff=6400`），计算可训练参数量；若参数为 float32，仅加载模型需多少内存？提交 1–2 句话。
2. 列出一次前向传播中的全部矩阵乘法，并计算总 FLOPs；输入长度等于 `context_length`。提交带说明的矩阵乘法列表与总 FLOPs。
3. 哪些部分消耗最多 FLOPs？提交 1–2 句话。
4. 对 GPT-2 small、medium、large 重复分析，并说明模型变大时各组件 FLOPs 占比如何变化。提交每个模型的组件占比及 1–2 句话说明。
5. 把 GPT-2 XL 上下文增至 16,384，分析总 FLOPs 与各组件相对占比的变化。提交 1–2 句话。

## 4. 训练 Transformer LM

### 4.1 交叉熵与困惑度

对位置 `i` 的 logits `o_i` 与目标 token `x_{i+1}`：

`ℓ_i = -log softmax(o_i)[x_{i+1}]`

整个训练集的损失是各 token 负对数似然的平均值。长度为 `m` 的序列对应困惑度：

`perplexity = exp((1/m) Σ_i ℓ_i)`。

#### 题目 `cross_entropy`：实现交叉熵

实现接收预测 logits 与目标 token 的函数。要求：

- 为数值稳定性减去最大元素；
- 尽可能在代数上约去 `log` 与 `exp`；
- 支持额外批式维度，并返回整个 batch 的平均值；词表维位于最后。

完成 `adapters.run_cross_entropy`，运行 `uv run pytest -k test_cross_entropy`。

### 4.2 SGD 与学习率

SGD 在每一步按 `θ_{t+1} = θ_t - α_t ∇L(θ_t; B_t)` 更新。优化器应继承 `torch.optim.Optimizer`，在 `__init__` 中注册参数和超参数，在 `step()` 中使用已计算的 `p.grad` 原地更新参数。

#### 题目 `learning_rate_tuning`：学习率调节（1 分）

对题目提供的 SGD 玩具示例，分别使用 `1e1`、`1e2`、`1e3`，训练 10 步。观察各学习率下损失下降更快、更慢还是发散。提交 1–2 句话。

### 4.3 AdamW

AdamW 为每个参数维护一阶、二阶矩估计，并把权重衰减与梯度更新解耦。常见 `(β1, β2)` 为 `(0.9, 0.999)`，大语言模型也常用 `(0.9, 0.95)`；`ε` 用于数值稳定。

#### 题目 `adamw`：实现 AdamW（2 分）

继承 `torch.optim.Optimizer`；构造函数接受学习率 `α`、`β`、`ε` 和权重衰减 `λ`。使用基类提供的 `self.state` 为各参数保存矩估计等状态。完成 `adapters.get_adamw_cls`，运行 `uv run pytest -k test_adamw`。

#### 题目 `adamwAccounting`：AdamW 训练资源核算（2 分）

假设所有 tensor 均为 float32：

1. 推导 AdamW 峰值内存，将参数、激活、梯度和优化器状态分别表示为 `batch_size` 及模型超参数的代数式，并给出总量。按题目列出的 RMSNorm、注意力、前馈层、最终 RMSNorm、输出 embedding 和交叉熵激活进行估算，令 `d_ff=4d_model`。
2. 代入 GPT-2 XL 配置，得到仅含 `batch_size` 的表达式；80 GB 内存下最大 batch size 是多少？
3. AdamW 单步需要多少 FLOPs？给出代数式和简短理由。
4. A100 float32 理论峰值为 19.5 TFLOP/s。假设 MFU 为 50%，单卡训练 GPT-2 XL，400K 步、batch size 1024 需要多少天？假设反向 FLOPs 是前向的两倍。

### 4.4 学习率调度

实现带 warmup 的余弦退火：

- `t < T_w`：学习率从 0 线性升至 `α_max`；
- `T_w ≤ t ≤ T_c`：按余弦曲线从 `α_max` 降至 `α_min`；
- `t > T_c`：保持 `α_min`。

#### 题目 `learning_rate_schedule`：实现带 warmup 的余弦学习率

函数接收 `t, α_max, α_min, T_w, T_c`，返回上述 `α_t`。完成 `adapters.get_lr_cosine_schedule`，运行 `uv run pytest -k test_get_lr_cosine_schedule`。

### 4.5 梯度裁剪

把全部参数梯度视作一个向量 `g`。若 `||g||₂ ≤ M` 则不变；否则按 `M/(||g||₂+ε)` 缩放，以防异常样本造成的大梯度破坏训练。

#### 题目 `gradient_clipping`：实现梯度裁剪（1 分）

函数接收参数列表和最大 L2 范数，原地修改各参数的梯度，使用 `ε=1e-6`。完成 `adapters.run_gradient_clipping`，运行 `uv run pytest -k test_gradient_clipping`。

## 5. 训练循环

### 5.1 数据加载

token 化后的数据是单一长序列 `x=(x_1,...,x_n)`，不同文档之间通常用 `<|endoftext|>` 分隔。数据加载器随机采样 `B` 个长度为 `m` 的输入序列，并配对相应的下一个 token 目标。固定长度避免 padding，也无需把整个数据集加载到内存。

#### 题目 `data_loading`：实现数据加载（2 分）

函数接收整数 NumPy 数组 `x`、`batch_size`、`context_length` 与设备字符串，返回输入与下一 token 目标两个 tensor。两者形状均为 `(batch_size, context_length)`，并放到指定设备。

完成 `adapters.run_get_batch`，运行 `uv run pytest -k test_get_batch`。

大数据集应通过 `np.memmap` 或 `np.load(..., mmap_mode='r')` 按需映射，dtype 必须与磁盘数组一致，并应检查 token ID 未超过词表范围。CPU 用 `cpu`，Apple Silicon 可用 `mps`。

### 5.2 Checkpoint

要恢复中断训练，checkpoint 至少要保存模型权重、优化器状态以及当前迭代数。模型和优化器均可通过 `state_dict()` / `load_state_dict()` 保存与恢复，整体对象可用 `torch.save` / `torch.load` 序列化。

#### 题目 `checkpointing`：实现 checkpoint（1 分）

实现：

- `save_checkpoint(model, optimizer, iteration, out)`：把三者状态写入路径或二进制文件对象；
- `load_checkpoint(src, model, optimizer)`：恢复模型和优化器并返回保存的迭代数。

完成 `adapters.run_save_checkpoint` 和 `adapters.run_load_checkpoint`，运行 `uv run pytest -k test_checkpointing`。

### 5.3 整体训练

#### 题目 `training_together`：组合全部组件（4 分）

编写脚本，在用户提供的数据上训练模型。建议至少支持：

- 配置模型与优化器的各种超参数；
- 用 `np.memmap` 高效加载大型训练集和验证集；
- 把 checkpoint 写到用户指定位置；
- 定期记录训练和验证性能，可输出到控制台或 Weights & Biases 等服务。

## 6. 文本生成

每次解码时，给模型一个前缀 `x_{1:t}`，取最后位置 logits，通过 softmax 得到下一 token 分布，从中采样并追加到序列；遇到 `<|endoftext|>` 或达到用户设置的最大长度时停止。

温度缩放把 logits 除以 `τ` 后再 softmax；`τ→0` 时分布趋近于集中在最大 logit 上。Top-p（nucleus）采样只保留按概率从高到低累计达到阈值 `p` 的最小 token 集合，然后重新归一化并采样。

#### 题目 `decoding`：解码（3 分）

实现模型解码函数，建议支持：

- 对用户 prompt 生成补全，直到生成 `<|endoftext|>`；
- 控制最大生成 token 数；
- 在采样前应用用户指定温度；
- 支持用户指定阈值的 top-p 采样。

## 7. 实验

要定期评估验证损失，同时记录梯度步数与实际用时，以便提交学习曲线。可以使用 Weights & Biases 等实验记录工具。

#### 题目 `experiment_log`：实验记录（3 分）

为训练和评估代码建立实验跟踪设施，能按梯度步数和实际用时记录实验与损失曲线。提交记录设施代码，以及包含后续所有尝试的实验日志文档。

### 7.1 TinyStories 基准配置

- `vocab_size = 10000`；
- `context_length = 256`；
- `d_model = 512`；
- `d_ff = 1344`，约为 `(8/3)d_model` 且是 64 的倍数；
- RoPE `Θ = 10000`；
- 4 层、16 个头，约 17M 非 embedding 参数；
- 总处理 token 数约 327,680,000，即 `batch_size × total_steps × context_length` 约等于该值。

需要自行寻找合适的学习率、warmup、AdamW 的 `β1, β2, ε` 与权重衰减。正确且高效的实现使用上述配置在一张 H100 上约需 30–40 分钟；明显更慢时，应检查数据加载、checkpoint、验证损失计算和批处理是否成为瓶颈。

调试建议：先在单个 minibatch 上过拟合，正确实现应能迅速把训练损失降到接近 0；检查中间 tensor 形状；监控激活、权重和梯度范数，排查爆炸或消失。

#### 题目 `learning_rate`：调节学习率（3 分，约 4 H100 小时）

1. 扫描多个学习率，报告最终损失或发散情况。提交多条学习曲线，并解释搜索策略；模型在 TinyStories 上的逐 token 验证损失应不高于 1.45。
2. 研究“最佳学习率位于稳定性边缘”这一经验：学习率开始发散的位置与最佳学习率有什么关系？提交逐渐增大学习率的曲线，至少包含一次发散实验，并分析其与收敛速度的关系。

低资源配置：CPU/MPS 可把总处理 token 数降到 40,000,000，并把目标验证损失放宽到 2.00。题目给出的 M3 Max 参考为 `32×5000×256=40,960,000` token，CPU 约 1 小时 22 分钟、MPS 约 36 分钟，最终验证损失约 1.80。余弦衰减应在最后一步到达最小学习率。题目特别提醒其测试版本中不要在 MPS 上启用 TF32。

#### 题目 `batch_size_experiment`：改变 batch size（1 分，约 2 H100 小时）

从 batch size 1 一直试到 GPU 内存极限，中间至少包含 64、128 等典型值；必要时重新优化学习率。提交不同 batch size 的学习曲线，以及几句话分析 batch size 对训练的影响。

#### 题目 `generate`：生成文本（1 分）

用 decoder 和训练好的 checkpoint 生成文本，可调节温度、top-p 等参数。提交至少 256 个 token 的文本，或生成到第一个 `<|endoftext|>` 为止；简评流畅度，并列出至少两个影响输出好坏的因素。

### 7.2 消融与架构修改

#### 题目 `layer_norm_ablation`：移除 RMSNorm（1 分，约 1 H100 小时）

移除 Transformer 中全部 RMSNorm 后训练。观察在此前最佳学习率下会发生什么，以及降低学习率能否恢复稳定。提交移除 RMSNorm 的学习曲线、最佳学习率曲线，以及几句话评论 RMSNorm 的影响。

#### 题目 `pre_norm_ablation`：改为 post-norm（1 分，约 1 H100 小时）

把 pre-norm 实现改为 post-norm 并训练。提交 post-norm 与 pre-norm 的对比学习曲线。

#### 题目 `no_pos_emb`：实现 NoPE（1 分，约 1 H100 小时）

从使用 RoPE 的模型中完全移除位置编码并训练。提交 RoPE 与 NoPE 性能对比学习曲线。

#### 题目 `swiglu_ablation`：SwiGLU 对比 SiLU（1 分，约 1 H100 小时）

把 SwiGLU 前馈网络与不带 GLU 门控的 SiLU 前馈网络比较。SiLU 版本使用 `FFN(x)=W2 SiLU(W1x)`，并取 `d_ff=4d_model`，以大致匹配 SwiGLU 的参数量。提交参数量近似匹配的两条学习曲线，并用几句话讨论发现。

低 GPU 资源的在线学习者可继续在 TinyStories 上进行这些修改，并用验证损失比较性能。

### 7.3 OpenWebText

OWT 比 TinyStories 更真实、复杂且多样。此实验可能需要重新调节学习率和 batch size。

#### 题目 `main_experiment`：OWT 实验（2 分，约 3 H100 小时）

使用与 TinyStories 相同的模型架构和总训练迭代数，在 OWT 上训练。提交：

- OWT 学习曲线，并描述其损失与 TinyStories 的差异以及应如何解释；
- 与 TinyStories 相同格式的生成文本，并讨论其流畅度；解释为何相同模型与计算预算下，OWT 输出质量更差。

### 7.4 自选修改与排行榜

排行榜规则：

- 单次提交在 H100 上最多运行 1.5 小时；
- 只能使用课程提供的 OWT 训练数据；
- 除此之外可以自由修改。

在完整 1.5 小时实验前，应先在 TinyStories 或 OWT 小子集上验证想法。小规模排行榜上有效的修改不一定能推广到更大规模预训练。

#### 题目 `leaderboard`：排行榜（6 分，约 10 H100 小时）

在上述规则下训练模型，目标是在 1.5 H100 小时内最小化验证损失。提交最终验证损失、横轴明确为实际用时且不超过 1.5 小时的学习曲线，以及所做修改的说明。结果至少应优于验证损失 5.0 的朴素基线。

## 建议完成顺序（非题目答案）

为避免后续错误叠加，可按依赖关系推进：

1. Unicode 书面题 → `train_bpe` → TinyStories/OWT BPE 实验；
2. `Tokenizer` → 分词器实验 → 把训练/验证数据保存为 token ID 数组；
3. `Linear`、`Embedding`、`RMSNorm`、SwiGLU、RoPE、softmax；
4. scaled attention → causal MHA → Transformer block → 完整 LM；
5. cross-entropy → AdamW → 调度器 → 梯度裁剪 → batch loader → checkpoint；
6. 先在单个 minibatch 上验证能够过拟合，再完成完整训练循环；
7. decoder → TinyStories 基准训练 → 学习率/batch size 实验；
8. 架构消融 → OWT → 可选排行榜。

每完成一个模块，先接好相应 adapter 并只运行该模块的测试；模块都通过后再做端到端训练。书面题中的计算、实验结果和分析应由你根据自己的推导与运行结果填写。
