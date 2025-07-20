# Decoder-Only Language Model (80M Parameters)

This project implements a custom decoder-only transformer language model in PyTorch. It is inspired by models like GPT, built from scratch for deep understanding and experimentation. The model supports sampling-based generation strategies such as **top-k**, **top-p (nucleus)** sampling with temperature scaling.

---

## 🚀 Overview

The goal of this project was to:
- Build a language model from scratch using PyTorch
- Implement core transformer architecture elements (embedding, multi-head attention, feed-forward layers)
- Support sampling-based text generation (instead of greedy decoding)
- Conduct experimentation with tokenization, training loops, and model evaluation

Our decoder-only model is trained to predict the next token in a sequence. It contains around 674,881 parameters, including token embeddings, multi-head self-attention layers, feedforward networks, and layer normalization components—making it compact yet expressive enough for experimentation. To run the model simply run the test file "python test_trainer_decoder_shakespeare.py" in the root folder. You can change the prompt also in the same file (line 96).

---

## 🧠 Model Architecture

- **Token + Positional Embedding**
- **N Layers** of:
  - Multi-head Self-Attention
  - Feed-Forward Network (MLP)
  - Residual Connections
  - Layer Normalization
- **Final Linear Head** for vocabulary logits

---

## 🧪 Test File Design

The `test_model.py` file was carefully crafted to:
- Load the model checkpoint
- Evaluate generation under **sampling conditions** (not greedy)
- Use a fixed prompt for consistent comparisons
- Measure inference time and output variation with `temperature`, `top_k`, and `top_p`

This design allows us to test creativity vs reliability of generation under varying sampling settings.

---

## 🎲 Sampling Techniques Implemented

Unlike traditional greedy decoding, we implemented **sampling-based generation**:

- To enhance the diversity and quality of generated text, we implemented probabilistic sampling strategies instead of traditional greedy decoding. Our generate() function uses the following techniques:
- Temperature scaling to control randomness.
- Top-k sampling, which restricts token selection to the k most probable tokens.
- Top-p (nucleus) sampling, which dynamically selects the smallest set of tokens whose cumulative probability exceeds a threshold p.
- These techniques allow the model to generate more natural, varied responses while maintaining relevance.

## 🐛 Issues Faced

- Tokenizer loading errors due to incorrect paths or missing files.
- High memory usage during model generation on CPU environments.
- Infinite loop during generation caused by faulty stopping condition.
- Model weights not matching architecture after manual edits.
- Difficulty validating generation due to randomness in sampling.
- Sampling logic errors: top_p and top_k clashed or misbehaved if both were used.
- Slow generation speed with large input sequence lengths.

## ✅ Issues Resolved

- Corrected the tokenizer paths and ensured all necessary files (tokenizer.json, tokenizer_config.json, vocab.json) were present.
- Optimized generation loop with @torch.no_grad() and in-place masking to improve speed and reduce memory load.
- Added a maximum length condition and correctly handled EOS tokens to stop generation.
- Made model class modular to easily load checkpoints and integrate sampling techniques.
- Implemented consistent sampling fallback (softmax + multinomial) for unpredictable token logits.
- Validated outputs using controlled temperature and seeds to verify randomness wasn't affecting logic.

## 🚧 Future Work

- Benchmark and compare decoding strategies (greedy, beam search, top-k, top-p) on BLEU / ROUGE scores.
- Add support for configurable repetition penalties and EOS token conditioning.
- Integrate with Hugging Face Trainer for easier training + inference pipelines.
- Explore quantization or LoRA to deploy the model on smaller devices.
- Package the model into a CLI or web interface for real-time use.
- Log and visualize attention scores or token-level probability graphs during generation.
