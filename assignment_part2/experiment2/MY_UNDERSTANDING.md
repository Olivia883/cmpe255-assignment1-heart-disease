# My Understanding

A language model learns next-token patterns from a sequence. The transformer
uses self-attention-style encoder blocks with causal masking so future tokens
cannot be used to predict the current token. Keeping the model small makes the
experiment practical on limited hardware.
