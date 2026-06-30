---
layout: post
title: Attention is an information-routing operator
date: 2026-06-29 10:00:00+0200
description: A conceptual walkthrough of attention as learned information routing. The post builds from tokens and Q/K/V projections to self-attention, multi-head attention, masking, positional encoding, and how the same ideas appear in robotics, VLMs, and VLA-style policies.
tags: attention network-architecture transformers self-attention
categories: paper-notes
thumbnail: assets/img/blog/attention/attention_multi.png
toc:
  beginning: true
related_posts: false
mermaid:
  enabled: true
  zoomable: false
---

Attention is a learned way of routing information between elements of a sequence or a set.
Given a collection of input vectors, attention asks: for each element, which other elements are relevant, and how much information should be gathered from them?

In the standard query-key-value view, each element produces a query, a key, and a value. The query is compared with keys from other elements to produce attention weights. These weights are then used to combine the corresponding values.
The result is a new, contextualized representation. Each element is no longer represented alone, but in relation to the elements it attended to.

In **NLP**, this means that a token can gather information from other tokens in the sentence. For example, a word may attend to another word that helps disambiguate its meaning, resolve a dependency, or provide context. Importantly, attention does not know word order by itself.
The order is provided through positional information.

In **robotics**, the same idea can be used more generally. Attention can learn which parts of a sensory, spatial, temporal, or body-structured representation should interact. For example, a policy may need to relate proprioceptive signals across joints, visual features across objects, or contact information across different parts of the body.

In this sense, attention is not just a language mechanism. It is a flexible operator for learning task-dependent interactions between structured features.

## Why attention was introduced

Before the Transformer, many sequence models were based on RNN, LSTM, GRU, or CNN architectures. These models can process ordered data, but they have an important limitation: distant positions often interact only after information has passed through many computational steps.

In an RNN, for example, information from token 1 reaches token 100 only after passing through many hidden-state updates. This creates a long path for both information flow and gradient propagation. LSTMs and GRUs improve this using gates, but they still process the sequence recurrently.

CNN-based sequence models have a different limitation. A small convolutional kernel only sees a local neighborhood. To connect far-apart positions, the model needs many layers, larger kernels, or dilation. In other words, distance in the input sequence is reduced only gradually through depth.

Self-attention changes this interaction pattern. In one self-attention layer, every position can directly compare itself with every other position and gather information from the relevant ones. This means that the computational path between two distant positions can be reduced to a single attention step {% cite vaswani2017attention --file blogs_attention %}.

This is one of the key ideas behind the Transformer: instead of forcing information to move sequentially through hidden states, or locally through convolutional neighborhoods, self-attention allows long-range interactions to be represented directly.

{% include figure.liquid loading="eager" path="assets/img/blog/attention/attention_multi.png" class="img-fluid rounded z-depth-1" zoomable=true %}

## Breaking down attention

A token, image patch, object feature, robot-state feature, or action token can be interpreted as asking: “Which other representations are useful for updating me, and how strongly should I mix them in?”

Example:

- NLP: “it” attends to the noun it refers to.
- Vision: a gripper patch attends to a mug-handle patch.
- Robotics: a future action token attends to the current object pose and previous action tokens.
- Diffusion policy: a noisy action sequence attends to visual observation embeddings.

### Tokens and embeddings

In NLP, a sentence is first split into **tokens**. Tokens are not always full words; they can be words, subwords, or punctuation pieces. For example:

```text
"The robot picked up the cup" → ["The", "robot", "picked", "up", "the", "cup"]
```

or with subword tokenization:

```text
"manipulation" → ["man", "ip", "ulation"]
```

Each token is then converted into a **vector embedding**. Practically, this is done using a learned embedding table. If the vocabulary has 50,000 tokens and each embedding has size 512, then the embedding table is of size $50,000 \times 512$.

Each token ID selects one row from this table.

```text
token "robot" → token id 1842 → embedding_table[1842] → vector of length 512
```

So a sentence becomes a sequence of vectors of tokens, `X_text = ["The", "robot", "picked", "up", "the", "cup"]`
is converted into embeddings `X_embd = [x_1, x_2, x_3, x_4, x_5, x_6]`
where each $x_i \in \mathbb{R}^{\text{d_model}}$, and $\text{d_model}$ is the embedding length. In this example, $\text{d_model} = 512$.

These embeddings can be learned during training, just like CNN filters. Initially they may be random; during training, the model adjusts them so that tokens used in similar contexts get useful vector representations.

### What are Q, K, V?

Once each token has an embedding vector, the Transformer computes three new representations from it: a query, a key, and a value.

For a sequence of token embeddings $X$, the model computes:

\begin{equation}
Q = X W_Q, \quad
K = X W_K, \quad
V = X W_V
\end{equation}

Here, $W_Q$, $W_K$, and $W_V$ are learned weight matrices. They are not manually designed; they are updated during training through the final task loss.

The attention operation is then:

\begin{equation}
\mathrm{Attention}(Q, K, V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
\end{equation}

For one token embedding $x_i \in \mathbb{R}^{d_{\mathrm{model}}}$, the corresponding query, key, and value are:

\begin{equation}
q_i = x_i W_Q, \quad
k_i = x_i W_K, \quad
v_i = x_i W_V
\end{equation}

So Q, K, and V are computed using the token embeddings, while the weight matrices that compute them are learned during training.
For a single attention head, the projection matrices have shapes such as:

- $W_Q \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$
- $W_K \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$
- $W_V \in \mathbb{R}^{d_{\mathrm{model}} \times d_v}$
  Often $d_v = d_k$, but conceptually the value dimension can be written separately.
  In many Transformer implementations, all heads are projected together, so the combined
  matrix is often written as $W_Q \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{model}}}$,
  then reshaped into multiple heads.

#### Intuition for Q, K, V

To understand Q, K, V, we can think of token embedding $x_i$ as being projected into three different roles:

```text
q_i: token i as a requester
k_i: token i as something others can match against
v_i: token i as information/content to pass forward
```

The terminology is somewhat similar to retrieval systems: a query is matched against keys, and the corresponding values are retrieved. However, in attention these are not symbolic database fields. They are learned continuous vectors.

For a token $i$, its query $q_i$ is compared with the key $k_j$ of every token $j$. This gives a compatibility score between token $i$ and token $j$: how useful token $j$ might be for updating token $i$.

\begin{equation}
\alpha\_{ij}
=
\mathrm{softmax}\_j
\left(
\frac{q_i k_j^\top}{\sqrt{d_k}}
\right)
\end{equation}

The softmax turns these compatibility scores into attention weights. These weights say how strongly token $i$ should gather information from each token $j$.

The updated representation of token $i$ is then computed as a weighted mixture of the value vectors from all tokens:

\begin{equation}
z*i = \sum_j \alpha*{ij} v_j
\end{equation}

The matrix $QK^\top$ is usually not symmetric.
Queries and keys are produced by different learned projections.
Entry $(i,j)$ and entry $(j,i)$ measure the relation in two different directions.
Thus, these two scores do not need to be the same.

```text
Q and K decide where to look, with a sense of directionality.
The softmax decides how strong the response should be for a particular relationship.
V contains the content that a given token contributes if another token attends to it.
```

The output $z_i$ is a contextualized representation of token $i$.
It is no longer based only on $x_i$, but also on the other tokens that token $i$ attended to.

#### Attention in robotics, VLMs, and VLAs

The same mechanism applies beyond language.

A token could be:

- a word token, such as `"drawer"`
- an image patch token containing part of a scene
- an object token representing a detected mug, drawer, or block
- a robot-state token, such as gripper pose or joint state
- an action token representing a future action or action chunk

In a VLM, attention can route information between language and image tokens.
For example, a language token such as `"drawer"` may attend to image patch tokens.
If a patch contains drawer-like visual features, the query from the language token and the key from that image patch may have high compatibility, allowing visual information from that patch to influence the language-conditioned representation.

In VLA models, the model connects vision, language, and action.
It is important not to confuse the attention value $V$ with the action output. The value vector $v_j$ is the information passed through attention.
The action may be predicted later by the model, or represented as an action token, depending on the architecture.

For example, in a Transformer-based robot policy, an action token may attend to observation tokens:

```text
action token:
"What action should I perform?"

observation tokens:
"object pose"
"gripper state"
"language goal"
"image feature"
```

In this simplified view, the action token acts like the query. The observation tokens provide keys and values. Attention allows the action representation to gather relevant visual, proprioceptive, or language information before the model predicts or denoises an action.

The exact implementation can vary across architectures. Some models explicitly use action tokens, while others condition action prediction through cross-attention, concatenated token sequences, or diffusion-style denoising over action chunks.

## Transformer architecture

The original paper is about translating one sequence into another {% cite vaswani2017attention --file blogs_attention %}. They evaluate the model on WMT 2014 English-to-German and English-to-French translation tasks.
The model uses an encoder-decoder structure.

The encoder reads the input sequence and creates contextual representations. The decoder generates the output sequence one token at a time.

In the original Transformer:

```text
Encoder: input tokens → contextual input representations
Decoder: previous output tokens + encoder representations → next output token
```

This is NLP-specific, but the abstraction is general.

For robotics, the encoder can process observations, images, proprioception, language, or other information needed for a contextual world representation.
The decoder can generate action tokens or trajectory tokens for the next action / action sequence.
This encoder-decoder view is useful for interpreting many robotics models, especially when observations are encoded into a contextual representation and actions or trajectories are generated conditionally.
However, modern VLA and diffusion-policy architectures may implement this conditioning using different Transformer, diffusion, or cross-attention variants.

The encoder is repeated $N=6$ times in the original paper. Each encoder layer has:

- multi-head self-attention
- feed-forward network
- residual connection
- layer normalization

The decoder is also repeated $N=6$ times. Each decoder layer has:

- masked multi-head self-attention over previous output tokens
- encoder-decoder attention over encoder outputs
- feed-forward network
- residual connection
- layer normalization

{% include figure.liquid loading="eager" path="assets/img/blog/attention/transformer_blocks.png" class="img-fluid rounded z-depth-1" zoomable=true %}

### Multi-head attention

Multi-head attention runs several attention operations in parallel.
A single attention head can be thought of as one learned routing pattern.
Multi-head attention has several learned routing patterns in parallel.

Instead of using one large attention operation over the full representation, the Transformer projects the input into multiple smaller representation subspaces. Each head has its own learned projections for Q, K, and V.

Then each head computes attention independently:

```text
head_i = Attention(Q_i, K_i, V_i)
```

The outputs of all heads are concatenated and passed through another learned linear projection:

```text
MultiHead(X) = Concat(head_1, ..., head_h) W_O
```

In the original Transformer base model, $d_{\mathrm{model}} = 512$, heads $h = 8$, and $d_k = d_v = 64$.
So each head attends in a lower-dimensional subspace, and the concatenated output returns to the model dimension.
The original paper describes this as allowing the model to jointly attend to information from different representation subspaces at different positions.

This is helpful because, in **NLP**, different heads may learn patterns related to local context, long-distance dependencies, syntactic structure, or pronoun resolution. In **robotics**, different heads may learn relations between object features, gripper state, contact-relevant visual features, robot-state tokens, language instructions, and future action tokens.

A useful caution is that heads are learned, not manually assigned. One head is not guaranteed to mean “object pose” or “syntax.”
Some heads may become interpretable, while others may overlap or support more distributed computations.

### Masked attention

In the Transformer, self-attention is masked in the decoder so that the model cannot look into future output tokens.
_This preserves autoregressive generation._
The model should not see the future word before predicting it.

In robotics, masking appears when predicting action sequences causally:
the token representing $a_t$ can attend to tokens representing $a_0 ... a_t$
but not future tokens such as $a_{t+1}$
However, not all robotics models need causal masking.
Diffusion models often denoise entire action chunks jointly, so they may use _bidirectional attention over action tokens depending on architecture_.

### Positional encoding

Attention by itself does not know order.

If we give tokens as a set, attention can compare all tokens, but it does not automatically know the sequence order.
For instance, token 1 comes before token 2 in language.

So the Transformer adds positional encodings to token embeddings.

In the original paper, sinusoidal encodings are used and position information is added to the embedding.
This gives each token a sense of where it is in the sequence.
This is beneficial when sequence or order of information matters.

Example:

```text
"robot grabs cup"
```

is different from:

```text
"cup grabs robot"
```

In robotics, position can mean:

- time index in an action sequence
- image patch location
- spatial position
- trajectory step
- order of proprioceptive history
- diffusion timestep, although diffusion timestep is usually represented separately

For diffusion policies, temporal position matters because action sequence order matters.

### Feed-forward network

After attention, each token goes through a feed-forward network.
The feed-forward network is applied position-wise, independently to each token position.
If the sequence representation is a matrix $X \in \mathbb{R}^{n \times d_{\mathrm{model}}}$, then each row $x_i$ is passed through the same MLP:

$$
\mathrm{FFN}(x_i)
=
\max(0, x_i W_1 + b_1)W_2 + b_2
$$

The feed-forward network then transforms the feature representation inside each token after attention has gathered contextual information.
In the original Transformer base model, the input and output dimension of this feed-forward block is $d_{\mathrm{model}} = 512$, while the inner hidden dimension is $d_{\mathrm{ff}} = 2048$.

### Residual connection and layer normalization

Each sub-layer is wrapped as:

```text
LayerNorm(x + Sublayer(x))
```

This means the _model does not replace the representation completely, it updates it._
This is similar to a residual connection in ResNet.

Thus each sublayer learns a modification:

```text
new representation = old representation + learned update
```

This helps optimization and allows deeper stacks. This is important because modern architectures often combine:

- residual blocks
- attention blocks
- normalization
- feed-forward transformations

## Retrieval summary

```text
Token:
one unit of representation, such as a word piece, image patch, object feature, robot-state token, or action token

Embedding:
a vector representation of that token

Q, K, V:
learned projections of token embeddings

Q:
query — what this token is looking for

K:
key — what this token exposes for matching

V:
value — the content this token can pass forward

QKᵀ:
pairwise query-key compatibility scores

A = softmax(QKᵀ / √d_k):
normalized attention weights

Z = AV:
the updated representation of token i after mixing information from the tokens it attends to
```

Attention therefore turns each token representation into a contextualized representation.
A token is updated not only from its own embedding, but also from the other tokens it attends to.

---

<h2>References</h2>

---

{% bibliography --file blogs_attention --cited_in_order --template bib_blog --group_by none %}
