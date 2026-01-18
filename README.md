# ToyLLM: Modern Transformer with KV Cache Compression

[![JAX](https://img.shields.io/badge/JAX-Latest-orange.svg)](https://github.com/google/jax)
[![Equinox](https://img.shields.io/badge/Equinox-DL%20Framework-blue.svg)](https://github.com/patrick-kidger/equinox)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An educational implementation of a GPT-style decoder-only transformer featuring modern architectural improvements and state-of-the-art KV cache compression techniques. Built with JAX and Equinox for high-performance deep learning research.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [KV Cache Compression](#kv-cache-compression)
- [Results](#results)
- [Project Structure](#project-structure)
- [References](#references)
- [Author](#author)

## 🎯 Overview

This project demonstrates the implementation of a modern transformer language model with a focus on:

1. **Modern Architectural Improvements**: RoPE, RMSNorm, GQA, SwiGLU, and Weight Tying
2. **Efficient KV Cache Management**: Standard autoregressive caching for fast generation
3. **SnapKV Compression**: Adaptive cache compression reducing memory by up to 70% while maintaining quality

The codebase is designed for educational purposes, making complex concepts accessible through clear implementations and extensive documentation.

## ✨ Key Features

### Modern Architecture Components

- **RoPE (Rotary Position Embeddings)**: Position encoding via complex vector rotation for better extrapolation
- **RMSNorm**: Efficient normalization replacing LayerNorm
- **GQA (Grouped-Query Attention)**: 50% KV cache reduction with minimal quality loss
- **SwiGLU Activation**: Gated activation function used in LLaMA and other SOTA models
- **Weight Tying**: Shared embeddings between input and output layers

### Advanced Caching Techniques

- **Standard KV Cache**: O(N) generation complexity vs O(N²) without caching
- **SnapKV Compression**: Adaptive token selection based on attention patterns
  - Attention Sinks: Preserve initial tokens for distribution stability
  - Heavy Hitters: Dynamic selection of important historical tokens
  - Local Window: Keep recent context for coherence

### Training & Evaluation

- **AdamW Optimizer**: With learning rate warmup and cosine decay
- **Gradient Clipping**: Stable training for deep networks
- **Label Smoothing**: Regularization technique
- **Perplexity Evaluation**: Comprehensive quality metrics

## 🏗️ Architecture

### Model Configuration

```python
VOCAB_SIZE = 50257      # GPT-2 tokenizer
CONTEXT_LEN = 1024      # Maximum sequence length
EMBED_DIM = 64          # Embedding dimension (educational scale)
N_HEAD = 2              # Attention heads (4 KV heads with GQA)
N_LAYERS = 2            # Transformer blocks
```

### Architecture Comparison

| Component | Vanilla Transformer | ToyLLM (Modern) |
|-----------|-------------------|-----------------|
| Position Encoding | Learned/Sinusoidal | **RoPE** |
| Normalization | LayerNorm | **RMSNorm** |
| Attention | MHA (all heads) | **GQA** (fewer KV heads) |
| MLP Activation | GELU/ReLU | **SwiGLU** |
| Embeddings/Head | Separate | **Weight Tying** |

### Forward Pass Modes

1. **Parallel Mode (Training/Prefill)**: Process entire sequences with causal masking
2. **Autoregressive Mode (Generation)**: Single token at a time with KV cache reuse

## 📦 Installation

### Requirements

```bash
pip install jax jaxlib equinox optax
pip install datasets transformers
pip install matplotlib seaborn numpy
```

### JAX Installation (GPU Support)

For GPU acceleration:

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## 📊 Dataset

### WikiText-103

The project uses **WikiText-103**, a large-scale language modeling dataset with:

- **Total tokens**: ~103 million (training set)
- **Vocabulary**: 267,735 unique tokens (50,257 with GPT-2 tokenizer)
- **Articles**: ~28,000 Wikipedia verified articles
- **Domain**: Long-form text with proper document structure

### Data Preprocessing Pipeline

1. **Loading**: Via HuggingFace Datasets library
2. **Tokenization**: GPT-2 tokenizer for consistency
3. **Concatenation**: Flatten into continuous token stream
4. **Batching**: Reshape to `[num_batches, batch_size, seq_len]`
5. **Truncation**: Ensure divisibility

```python
data, tokenizer = load_and_prep_wikitext(
    split="validation",
    seq_len=512,
    batch_size=4
)
```

## 🚀 Usage

### 1. Model Initialization

```python
import jax
from model import ToyLLM

key = jax.random.PRNGKey(42)

model = ToyLLM(
    vocab_size=50257,
    context_len=1024,
    embed_dim=64,
    n_head=2,
    n_layers=2,
    key=key
)
```

### 2. Training

```python
trained_model = train_model(
    model,
    data,
    batch_size=32,
    seq_len=128,
    steps=1000,
    learning_rate=3e-4,
    warmup_steps=50,
    weight_decay=0.1,
    grad_clip=1.0
)
```

### 3. Generation with Standard KV Cache

```python
# Verify cache correctness
verify_kv_cache(model, seq_len=50)
```

### 4. Evaluation with SnapKV Compression

```python
ppl, history, losses = evaluate_with_compression(
    model,
    data,
    seq_len=200,
    budget=64,        # Cache size limit
    sink_size=4,      # Attention sinks
    window_size=16    # Local window
)
```

### 5. Performance Benchmarking

```python
run_benchmark_with_compression_fixed(
    model,
    max_seq_len=200,
    step_interval=5,
    budget=64
)
```

## 🗜️ KV Cache Compression

### The Problem

Standard KV cache grows linearly with sequence length:
- At 1K tokens: ~1 MB per batch
- At 100K tokens: ~100 MB per batch
- Memory becomes bottleneck for long contexts

### SnapKV Solution

**Three-Component Strategy:**

1. **Attention Sinks (4 tokens)**: Initial tokens that stabilize attention distribution
2. **Heavy Hitters (dynamic)**: Important historical tokens selected via attention voting
3. **Local Window (16 tokens)**: Recent context for coherence

**Selection Algorithm:**

```
Step 1: Recent window votes on historical importance
Step 2: Max pooling over attention scores (cluster-aware)
Step 3: Mask attention sinks (added separately)
Step 4: Select top-K tokens with highest pooled scores
Step 5: Concatenate [sinks + heavy_hitters + window]
```

### Compression Configuration

| Budget | Compression | PPL Δ | Use Case |
|--------|-------------|-------|----------|
| Full | 0% | Baseline | Training, short contexts |
| 128 | 50-75% | <2% | Long documents |
| 64 | 75-85% | 2-5% | Very long contexts |
| 32 | 85-95% | 5-15% | Extreme compression |

## 📈 Results

### KV Cache Verification

Standard KV cache produces **numerically identical outputs** to full recomputation:
- Max difference: < 1e-6 (floating point precision)
- Speed: ~1000× faster at 1024 tokens

### SnapKV Compression Performance

#### Memory Efficiency
- **Standard cache**: Linear growth (0.3 MB at 200 tokens)
- **Compressed cache**: Capped at 0.06 MB
- **Memory saving**: ~70% reduction

#### Latency
- **No cache baseline**: Linear degradation with sequence length
- **Standard cache**: Constant time per token (~0.5ms)
- **Compressed cache**: Constant time per token (~0.6ms, <20% overhead)
- **Speedup**: 150× faster than baseline

#### Quality Metrics
- **Perplexity**: Budget 64 maintains <5% degradation
- **Generation stability**: Moving average loss remains stable around 10.0
- **Token retention**: Adaptive selection preserves semantically important tokens

### Compression Dashboard

The project includes comprehensive visualization tools:

1. **Generation Stability**: Cross-entropy loss over generation steps
2. **Cache Retention Map**: Heatmap showing which tokens survive compression
3. **Token Distribution**: Histogram of retained token positions
4. **Performance Metrics**: Latency, memory, speedup, and compression ratio

## 📁 Project Structure

```
Main.ipynb
├── Imports & Setup
├── Dataset Explanation & Loading
│   └── WikiText-103 preprocessing
├── Architecture Components
│   ├── RoPE (Rotary Position Embeddings)
│   ├── RMSNorm
│   ├── GQA Attention
│   ├── SwiGLU MLP
│   └── ToyLLM Model Class
├── KV Cache Implementation
│   ├── Cache verification
│   └── Autoregressive generation
├── SnapKV Compression
│   ├── Compression algorithm
│   ├── Position tracking
│   └── Visualization tools
├── Training Pipeline
│   ├── AdamW optimizer
│   ├── Learning rate scheduling
│   └── Metrics logging
└── Benchmarking & Evaluation
    ├── Performance analysis
    └── Scientific plots
```

## 🔬 Technical Details

### JAX Optimizations

- **JIT Compilation**: All core functions are JIT-compiled for performance
- **VMAP**: Automatic batching for parallel processing
- **Functional Programming**: Pure functions enable automatic differentiation

### Attention Mechanism

```python
# Standard multi-head attention with causal masking
scores = (Q @ K.T) / sqrt(head_dim) + mask
attn_weights = softmax(scores)
output = attn_weights @ V
```

### Compression Integration

- **Transparent**: Works with existing model without modifications
- **Adaptive**: Heavy hitters change based on generation context
- **Robust**: Handles both prefill and generation phases

## 📚 References

### Papers

1. **RoPE**: Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
2. **GQA**: Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models" (2023)
3. **SwiGLU**: Shazeer "GLU Variants Improve Transformer" (2020)
4. **SnapKV**: Li et al. "SnapKV: LLM Knows What You are Looking for Before Generation" (2024) - arXiv:2404.14469
5. **StreamingLLM**: Xiao et al. "Efficient Streaming Language Models with Attention Sinks" (2023)

### Frameworks

- [JAX](https://github.com/google/jax): High-performance numerical computing
- [Equinox](https://github.com/patrick-kidger/equinox): Elegant neural networks in JAX
- [Optax](https://github.com/deepmind/optax): Gradient processing and optimization
- [HuggingFace Datasets](https://github.com/huggingface/datasets): Dataset loading and processing

## 👤 Author

**Matteo Sorrentini** (ID: 2023085, Email: sorrentini.2023085@studenti.uniroma1.it)

Neural Networks Course Project - Data Science Master's Program

*University of Rome "La Sapienza"*

## 📄 License

This project is available for educational purposes. Feel free to use it for learning and research.

---

**Note**: This is an educational implementation. For production use, consider optimized libraries like vLLM, FlashAttention, or Transformers with quantization.
