MiniGPT

A small GPT-style decoder-only language model built from scratch in PyTorch to understand how modern language models work internally.

Project Status: 🚧 Work in Progress

This project is being developed step by step. The main goal is not just to use an existing LLM, but to understand and implement the core components behind a GPT-style model.

What I am Building

The current architecture follows a decoder-only GPT-style design:

Text
  ↓
GPT-2 Tokenizer
  ↓
Token IDs
  ↓
Token + Positional Embeddings
  ↓
Causal Self-Attention
  ↓
Add & LayerNorm
  ↓
Feed Forward Network
  ↓
Add & LayerNorm
  ↓
Multiple Decoder Blocks
  ↓
Final LayerNorm
  ↓
Language Modeling Head
  ↓
Next-Token Prediction

The model is designed for next-token prediction: given a sequence of tokens, it learns to predict what token should come next.

Current Configuration

The current model is intentionally small compared with production LLMs, but it is still large enough to make training computationally expensive on a normal laptop.

Tokenizer: GPT-2 BPE via tiktoken

Vocabulary size: 50,257

Context length: 512 tokens

Embedding dimension: 384

Attention heads: 6

Decoder layers: 6

Head dimension: 64

Feed-forward dimension: 1,536

Activation: GELU

Dropout: 0.1

Optimizer: AdamW

Learning-rate warmup + cosine decay

Gradient clipping

Mixed precision on CUDA

Dataset: Tiny Shakespeare

Why the Project Is Still in Progress

The architecture and training pipeline are being implemented and tested, but the model is not yet a completely finished conversational GPT.

One of the main reasons is the number of parameters and the computational cost involved in training a transformer model. Even though this is a relatively small model, training it properly requires significant GPU time.

The current focus is therefore on:

Making sure every component works correctly.

Testing the complete forward pass.

Making the training pipeline stable.

Training on a GPU through Google Colab.

Monitoring training and validation loss.

Improving generation quality.

Completing the end-to-end conversational workflow.

The goal is to build a working model first and then improve its quality rather than claiming that the model is already a fully capable ChatGPT-like system.

Project Progress

Completed / Implemented

Project structure

GPT-style configuration

GPT-2 tokenizer integration

Dataset and next-token target generation

Token embeddings

Learned positional embeddings

Causal self-attention implementation

Multi-head attention

Feed-forward network

Decoder block structure

Residual connections

Layer normalization

GPT model architecture

Weight tying between token embedding and LM head

Language-modeling head

Cross-entropy training objective

AdamW optimizer

Gradient clipping

Learning-rate warmup

Cosine learning-rate decay

Periodic evaluation

Periodic checkpoint saving

Best-model checkpoint saving

CUDA / mixed-precision training support

Basic component test files

In Progress

Full training run on Google Colab GPU

Validate training and validation loss behavior

Improve generated text quality

Complete robust text-generation pipeline

Resume training reliably from checkpoints

End-to-end conversational testing

Final documentation and benchmarking

Testing

The project contains tests for the main components:

tests/
├── test_config.py
├── test_tokenizer.py
├── test_dataset.py
├── test_embeddings.py
├── test_decoder.py
├── test_model.py
└── test_summary.py

The tests are used to verify configuration, tokenization, dataset shifting, tensor shapes, decoder behavior, model forward passes, and parameter counts before starting a full training run.

Training

The model is trained using Tiny Shakespeare for the current experiment.

The training pipeline is:

Tiny Shakespeare
      ↓
GPT-2 Tokenization
      ↓
Training / Validation Data
      ↓
DataLoader
      ↓
GPTModel
      ↓
Logits
      ↓
Cross-Entropy Loss
      ↓
Backpropagation
      ↓
Gradient Clipping
      ↓
AdamW
      ↓
Learning-Rate Scheduler
      ↓
Weight Update

For GPU training, Google Colab is being used because the model is computationally heavier than a typical small Python project.

Generation

After training, the model can be used for autoregressive generation:

Prompt
  ↓
Tokenization
  ↓
Model
  ↓
Next-token probabilities
  ↓
Temperature / Top-K / Top-P sampling
  ↓
Next token
  ↓
Append token to context
  ↓
Repeat
  ↓
Generated text

The current generation system is being developed alongside the training pipeline.

Important Limitation

This project should not currently be considered a production-ready chatbot or a replacement for large pretrained language models.

The purpose of MiniGPT is primarily educational and experimental: to understand the architecture, data flow, training process, optimization, and inference process of GPT-style language models by implementing the major components directly.

Hardware

Training is being tested on Google Colab GPU, including NVIDIA Tesla T4 environments.

A normal laptop can be used for development and testing, but full training can take considerably longer because transformer training is computationally expensive.

Tech Stack

Python

PyTorch

Tiktoken

NumPy

tqdm

Google Colab

Git / GitHub

Repository Structure

MiniGPT/
│
├── data/
│   └── tiny_shakespeare.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── feed_forward.py
│   ├── decoder.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   └── utils.py
│
├── tests/
│   ├── test_config.py
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   ├── test_embeddings.py
│   ├── test_decoder.py
│   ├── test_model.py
│   └── test_summary.py
│
├── checkpoints/
├── logs/
├── requirements.txt
├── .gitignore
└── README.md

Future Improvements

Once the current training pipeline is stable, the next improvements will focus on:

Better training stability

More training data

Better evaluation

Improved generation quality

Checkpoint resume support

More efficient training

Better conversational behavior

More detailed model benchmarking

MiniGPT is a learning project built to understand what happens inside a GPT-style language model — from raw text and tokenization all the way to training and next-token generation.
