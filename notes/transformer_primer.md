Transformer -

Essentially allows us to iteratively construct text using prediction.
We start with a word, and try to predict the next best word until we have a fully constructed response.

Here are the details -

Tokens are subwords learned from training data via byte-pair encoding (BPE)
Embeddings -    Each token is represented by a vector (the what) and has a position (the where)
Attention -     This is the brain, how are the words related to each other. We have the query, key and value for each token here. 
                Think of query as I am looking for x, key as I am an expert in y, value as more information to share when a relevant match is found.
                So each query tries to match to all the keys to find the best match and then share the value when a match is found. Heads are involved which give more context to the token.
MLP -           The thinking happens in this step, each token sits with all the collected information from the previous step and figures out which
                information makes the best sense -> After attention, each token's vector now contains a weighted mixture of information from all prior positions. The MLP then processes that mixture token-by-token, independently. Everyone works independently in this step -> The MLP applies the same transformation independently to each position's vector — there's no cross-token interaction in the MLP.
LayerNorm -     This ensures normalization across the matrix, so the meaning of a token is not lost by the value being too big or too small. Mean is set to 0
                and the variance is set to 1.
Loop -          Depth loop (inside the model, one forward pass): input flows through N transformer blocks (e.g., N=12 for GPT-2 small). This is the stack. 
                Each block does attention + MLP. One forward pass = one walk through all N blocks.
                Generation loop (outside the model, autoregression): at inference time: predict next token → append → run forward pass again → predict → append → ... until done.
                Distinguish these. "Stacking" for depth (block 1 → block 2 → ... → block N), "generation loop" for autoregression.

Tracing the shapes through the process -
(B, T) → (B, T, n_embd) → (B, n_head, T, head_size) → (B, T, n_embd) → (B, T, vocab_size)

Multi-head attention purpose
Why split into multiple heads? Each head can specialize. One head learns "attend to the previous noun"; another learns "attend to the matching parenthesis"; another learns "attend to the verb." More parallel patterns. The cost: each head gets fewer dimensions (head_size = n_embd / n_head).
"Multiple heads let attention specialize on different patterns in parallel."

Causal mask
This is why GPT can do next-word prediction. During training, position i should only see positions 0..i. We zero out (set to -inf before softmax) the upper-triangular half of the score matrix. Without this, the model would "cheat" by peeking at the future.
"Causal mask: position i only sees 0..i, preventing the model from peeking at future tokens."

Residual connections
Each block does x = x + attention(LayerNorm(x)) and x = x + MLP(LayerNorm(x)). The x + is the residual. Critical because gradients need an unobstructed path back through the network during backprop. Without residuals, you can't train models with 12+ blocks.
"Residual connections (x + ...) let gradients flow back through deep stacks of blocks."

Output projection (lm_head)
You implicitly cover this with "predict the next best word," but the mechanism is worth naming: a final linear layer maps (B, T, n_embd) → (B, T, vocab_size). Each position outputs a score for every token in the vocab. Softmax gives probabilities.
"Final linear projection (lm_head) maps each position's vector to a score over the full vocabulary; softmax → next-token probabilities."

One nanoGPT-vs-paper difference
Pre-norm: nanoGPT applies LayerNorm before attention/MLP; original paper applied it after the residual.
Weight tying: nanoGPT shares weights between input embedding and output projection. Saves parameters.
FlashAttention via F.scaled_dot_product_attention: PyTorch ≥2.0 routes attention to FlashAttention automatically. Not in the original paper.


# Transformer — primer notes

A transformer predicts the next token given a sequence of prior tokens.
Generate text by repeating: predict → append → predict.

## The pipeline (one forward pass)
1. **Tokenize**: text → subword token IDs (BPE; ~50K-vocab for GPT-2).
2. **Embed**: each token ID → vector via lookup. Add a learned position
   vector so the model knows where each token sits.
3. **Stack of N transformer blocks** (N=12 for GPT-2 small). Each block:
   - **Attention** (with multiple heads): each token computes Q, K, V
     from itself. Q matches against all K via dot product → attention
     weights → weighted sum of V vectors. Causal mask hides future
     positions. Multiple heads let the model attend on different
     patterns in parallel.
   - **MLP**: per-token, point-wise: linear up → GELU → linear down.
     No cross-token interaction here — attention already mixed.
   - **LayerNorm + residual** wrap each sub-block.
4. **Final LayerNorm**, then **lm_head** linear projection maps each
   token's vector → vocab-sized score → softmax → probabilities.

## Two loops to distinguish
- **Depth loop**: input walks through all N blocks once per forward pass.
- **Generation loop**: at inference, run forward pass, sample next token,
  append, repeat.

## What nanoGPT does differently from the original paper
nanoGPT uses **pre-norm** (LayerNorm before attention/MLP, applied to
the input only); the paper used post-norm. Pre-norm is more stable at
depth, used by most modern LLMs.



Tokenization is the model's input encoding — text into subword IDs. Chunking is a RAG preprocessing decision — splitting long documents into retrievable passages before you embed and store them. They're independent concerns at different layers of the stack.