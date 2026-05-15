# MiniMind — 思政教育轻量级大语言模型

MiniMind 是一个从零实现的轻量级中文大语言模型，专为**思想政治教育**场景定制。模型参数量约 **25M**，支持从预训练到领域微调的完整训练流程，可直接在消费级显卡上运行。

## 项目背景

由沈阳师范大学 CS 团队开发，目标是打造一个扎根思政领域的轻量 AI 助手，能够准确回答马克思主义理论、党史国史、时事政策解读等思政相关问题。

> "CS 赋予我智慧，思政赋予我灵魂。"

## 模型架构

基于 Transformer Decoder-Only 架构，从零手写实现，不依赖 HuggingFace 模型代码：

| 特性 | 配置 |
|------|------|
| 隐藏层维度 | 768 |
| Transformer 层数 | 8 |
| 注意力头数 | 8 (GQA 4 KV heads) |
| 词表大小 | 6400 (BPE) |
| 总参数量 | ~25M |
| 最大上下文 | 32768 (RoPE + YaRN) |
| 激活函数 | SiLU (SwiGLU) |
| 归一化 | RMSNorm |
| 注意力 | Flash Attention / SDPA |
| 可选架构 | MoE (Mixture of Experts) |

### 与标准 LLaMA 风格的主要区别

- **词表极小** — 6400 tokens vs 常见 32k+，极大降低 Embedding 参数量，适合中文小模型
- **GQA 注意力** — 8 个 query 头 + 4 个 key/value 头，减少 KV Cache
- **YaRN 外推** — 支持推理时位置编码外推，突破训练长度限制
- **可选 MoE** — 支持混合专家架构，同等参数量下降低推理计算量

## 训练流程

```
Pre-training (from scratch)
    ↓
Full SFT (全量指令微调)
    ↓
LoRA (领域专项微调) → 思政领域适配
```

### 1. 预训练 (`trainer/train_pretrain.py`)

从头训练 MiniMind 基座模型，学习通用语言能力。

```bash
python trainer/train_pretrain.py \
    --epochs 2 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --max_seq_len 340 \
    --data_path ../dataset/pretrain_t2t_mini.jsonl
```

### 2. 全量 SFT (`trainer/train_full_sft.py`)

在预训练权重基础上进行指令微调，让模型学会对话格式和指令遵循。

```bash
python trainer/train_full_sft.py \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --from_weight pretrain \
    --data_path ../dataset/sft_t2t_mini.jsonl
```

### 3. LoRA 微调 (`trainer/train_lora.py`)

冻结主模型，仅训练低秩适配器，高效注入思政领域知识。

```bash
python trainer/train_lora.py \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 2e-4 \
    --lora_name lora_politics_identity \
    --from_weight full_sft \
    --data_path ../dataset/ideological_politics.jsonl
```

### 4. 分词器训练 (`trainer/train_tokenizer.py`)

基于 BPE 算法训练自定义分词器，含特殊 token（工具调用、思考标签、多模态占位符等）。

> ⚠️ 仅供学习参考，不建议重新训练，MiniMind 已自带分词器。

## 推理

```bash
python eval_llm.py \
    --weight full_sft \
    --lora_weight lora_politics_identity \
    --temperature 0.15 \
    --max_new_tokens 512
```

交互模式支持自动测试和手动输入两种方式。

## 项目结构

```
├── model/
│   ├── model_minimind.py    # MiniMind 模型定义（含 Config、Attention、MoE）
│   ├── model_lora.py        # LoRA 实现（apply/save/load/merge）
│   ├── tokenizer.json       # BPE 分词器
│   └── tokenizer_config.json
├── trainer/
│   ├── train_pretrain.py    # 预训练脚本
│   ├── train_full_sft.py    # 全量 SFT 脚本
│   ├── train_lora.py        # LoRA 微调脚本
│   ├── train_tokenizer.py   # 分词器训练脚本
│   └── trainer_utils.py     # 训练工具（DDP、checkpoint、学习率调度等）
├── out/
│   ├── pretrain_768.pth            # 预训练权重 (132MB)
│   ├── full_sft_768.pth            # SFT 权重 (132MB)
│   └── lora_politics_identity_768.pth  # LoRA 思政权重 (780KB)
├── eval_llm.py               # 推理/对话脚本
├── ideological_politics.jsonl # 思政领域训练数据
└── dataset/                   # 训练数据目录（未包含在仓库中）
```

## 技术亮点

- **全手写实现** — 模型代码不依赖 HuggingFace 模型类，从 Config 到 Attention 到 Generation 完全自建
- **与现代架构对齐** — RoPE、GQA、SwiGLU、RMSNorm、Flash Attention 等 LLaMA 系列核心理念完整实现
- **分布式训练** — 支持 DDP 多卡训练、梯度累积、混合精度、断点续训
- **LoRA 高效微调** — 仅训练 ~0.02M 参数即可注入领域知识，权重文件仅 780KB
- **YaRN 长度外推** — 支持推理时扩展至训练长度 16 倍以上
- **Token 级流式解码** — 支持字节缓冲的流式输出

## 示例问答

```
问：四个自信是什么？
答：四个自信是道路自信、理论自信、制度自信、文化自信。
    道路自信是对中国特色社会主义道路的自信；
    理论自信是对中国特色社会主义理论体系的自信；
    制度自信是对中国特色社会主义制度的自信；
    文化自信是对中国特色社会主义文化的自信。
    坚定四个自信是坚持和发展中国特色社会主义的必然要求。

问：绿水青山就是金山银山是啥？
答：这是习近平生态文明思想的核心理念。它阐明了生态
    环境保护与经济发展的辩证关系。绿水青山既是自然财富、
    生态财富，又是社会财富、经济财富。必须坚持生态优先、
    绿色发展，将生态优势转化为经济优势。
```

## 依赖环境

- Python 3.10+
- PyTorch 2.0+
- Transformers（仅用于分词器加载）
- tokenizers（分词器训练）

```bash
pip install torch transformers tokenizers
```

## 致谢

本项目模型架构参考了 [MiniMind 开源项目](https://github.com/jingyaogong/minimind)，在此表示感谢。
