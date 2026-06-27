### LANGUAGE MODEL
Estimates the probability of a token or sequence of tokens occurring within a longer sequence of token.
1. A token could be a word, a sub word or even a single character.
### TOKENIZATION & VOCABULARIES
When building Generative AI applications, everything begins and ends with how text is converted into data that a machine can compute.

### The Core Problem: Why We Don't Use Characters or Words
If you design an AI model, you face foundational choice: What is the atomic unit of text?
#### APROACH A: CHARACTER-LEVEL TOKENIZATION
Every letter, number, and punctuation mark is a token ("c", "a").
##### The flaw:
1. Character carry zero semantic meaning on their own. 
2. Model has to expand massive computational power just learning that h + o + u + s + e
mean a physical dwelling.
3. If sequence length becomes massive, quickly exhausting the model's memory limits.

#### APPROACH B : WORD-LEVEL TOKENIZATION
Every unique word is a token ("cat", "running")
##### The flaw:
1. English vocabulary has thousands of words.
2. If you include plurals, tenses and typos ("running", "runs", "runng"), your model's dictionary scales to millions of items.

#### THE SOLUTION : SUB-WORD TOKENIZATION
Modern LLMs use Sub-word Tokenization. 
2. Frequent words stay whole (e.g. "the", "python"), while rare or complex words are broken down into pieces (e.g. "tokenization" becomes ["token", "ization"]). 

### 2. The Dominant Tokenization Algorithms

Different model families use different algorithmic frameworks to build their vocabulary during their initial training phase:

| **Algorithm**                | **Primary Ecosystem**        | **Key Characteristic**                                                                                                                                          |
| ---------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Byte-Pair Encoding (BPE)** | GPT series, Llama, Anthropic | Starts with individual characters and merges the most frequently adjacent pairs iteratively.                                                                    |
| **WordPiece**                | BERT, Google's older models  | Similar to BPE, but chooses merges based on maximizing the likelihood of the training data according to a language model.                                       |
| **Byte-level BPE (BBPE)**    | Modern LLMs (GPT-4, Gemini)  | Instead of characters, it starts with raw **bytes** ($0$ to $255$). This ensures it can parse _any_ text, emoji, or non-English script without throwing errors. |

### 3. Vocabulary Size ($|V|$) and Trade-offs

The **Vocabulary ($V$)** is the fixed set of all unique tokens the model knows.

- **Small Vocab ($\approx 32,000$ tokens, e.g., Llama 1):** Efficient memory footprint at the base matrix level, but non-English text or code is broken into many tiny chunks, inflating the token count per prompt.
- **Large Vocab ($\approx 100k - 256k$ tokens, e.g., Llama 3, Gemini):** The model can handle multiple languages and structured code highly efficiently (fewer tokens per sentence), but the initial layer of the model requires more parameter overhead.

### . What is an Embedding?

An embedding is a vector (a list of floating-point decimal numbers) that represents the semantic meaning of a piece of text.

Instead of treating a word as an isolated point, an embedding model maps text into a **high-dimensional vector space**.

- A typical embedding vector might have **768**, **1536**, or even **3072** dimensions.
    
- Each dimension represents a subtle, latent semantic feature or concept (e.g., "animate vs. inanimate", "tense", "gender", "technicality"), though these features are learned automatically by the model and aren't easily readable by humans.
    

> **The Fundamental Rule of Embeddings:** > Text snippets with similar meanings are placed closer together in this mathematical space. For example, the vector for `"king"` will be mathematically very close to `"queen"`, and `"python"` (the language) will sit closer to `"code"` than to `"snake"`.

### 2. How the Vector Space Handles Meaning

Because embeddings are numerical vectors, you can perform spatial math on language.
#### Classic Vector Analogy:

If you take the vector for **"King"**, subtract the vector for **"Man"**, and add the vector for **"Woman"**, the resulting vector lands remarkably close to the coordinates for **"Queen"**:

$$\vec{v}_{\text{King}} - \vec{v}_{\text{Man}} + \vec{v}_{\text{Woman}} \approx \vec{v}_{\text{Queen}}$$

#### Dense vs. Sparse Vectors

- **Sparse Vectors (Older/Traditional IR):** Think BM25 or TF-IDF. These match exact words. If your document says "automobile" and the user searches "car", a sparse vector approach won't inherently match them unless manually linked.
    
- **Dense Vectors (Modern Embeddings):** Every position in the vector is a non-zero number. They capture _concepts_ over exact string matches.

### 3. Measuring Similarity: Distance Metrics

When building real GenAI systems (like RAG), you need a way to mathematically answer: _"How relevant is this document chunk to the user's query?"_ We use distance or similarity metrics to compare two vectors ($\vec{A}$ and $\vec{B}$).

#### A. Cosine Similarity

Measures the _angle_ between two vectors, completely ignoring their length (magnitude). This is the most popular metric for text because text length shouldn't drastically change the core meaning.

- Range: $-1$ to $1$ (where $1$ means completely identical direction).
    
- Formula:
    
    $$\text{Cosine Similarity} = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$
    

#### B. Dot Product (Inner Product)

Multiplies corresponding elements and sums them up. If your vectors are unit-normalized (length of 1), Dot Product is mathematically identical to Cosine Similarity but computes much faster.

- Formula:
    
    $$\vec{A} \cdot \vec{B} = \sum_{i=1}^{n} A_i B_i$$
    

#### C. Euclidean Distance ($L_2$ Distance)

Measures the straight-line distance between two points in space. It is highly sensitive to vector length.

- Formula:
    
    $$d(\vec{A}, \vec{B}) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$$

#### THE TRANSFORMER
Transformer is a neural network architecture that has fundamentally changed the approach to Artificial Intelligence. Transformer was first introduced in the seminal paper ["Attention is All You Need"](https://dl.acm.org/doi/10.5555/3295222.3295349 "ACM Digital Library") in 2017.

2. Fundamentally, text-generative Transformer models operate on the principle of **next-token prediction**: given a text prompt from the user, what is the _most probable next token (a word or part of a word)_ that will follow this input? The core innovation and power of Transformers lie in their use of self-attention mechanism, which allows them to process entire sequences and capture long-range dependencies more effectively than previous architectures.

# Transformer Architecture

Every text-generative Transformer consists of these **three key components**:

1. **Embedding**: Text input is divided into smaller units called tokens, which can be words or sub words. These tokens are converted into numerical vectors called embeddings, which capture the semantic meaning of words.
2. **Transformer Block** is the fundamental building block of the model that processes and transforms the input data. Each block includes:
    - **Attention Mechanism**, the core component of the Transformer block. It allows tokens to communicate with other tokens, capturing contextual information and relationships between words.
    - **MLP (Multilayer Perceptron) Layer**, a feed-forward network that operates on each token independently. While the goal of the attention layer is to route information between tokens, the goal of the MLP is to refine each token's representation.
3. **Output Probabilities**: The final linear and softmax layers transform the processed embeddings into probabilities, enabling the model to make predictions about the next token in a sequence.

## EMBEDDING
The prompt needs to be converted into a format that the model can understand and process.
- That is where embedding comes in: it transforms the text into a numerical representation that the model can work with. 
### STEPS TO CONVERT A PROMPT INTO EMBEDDING

### STEP1: TOKENIZATION

1. Tokenization is the process of breaking down the input text into smaller, more manageable pieces called tokens
2. The full vocabulary of tokens is decided before training the model: GPT-2's vocabulary has `50,257` unique tokens. Now that we split our input text into tokens with distinct IDs, we can obtain their vector representation from embeddings.

### STEP2 : TOKEN EMBEDDING

1. GPT-2 (small) represents each token in the vocabulary as a 768-dimensional vector; the dimension of the vector depends on the model.
2. These embedding vectors are stored in a matrix of shape `(50,257, 768)`, containing approximately 39 million parameters!
3. This extensive matrix allows the model to assign semantic meaning to each token, in the sense that tokens with similar usage or meaning in language are placed close together in this high-dimensional space, while dissimilar tokens are farther apart.

### Step 3. Positional Encoding

The Embedding layer also encodes information about each token's position in the input prompt. Different models use various methods for positional encoding. GPT-2 trains its own positional encoding matrix from scratch, integrating it directly into the training process.

### Step 4. Final Embedding

Finally, we sum the token and positional encodings to get the final embedding representation. This combined representation captures both the semantic meaning of the tokens and their position in the input sequence.

# TRANSFORMER BLOCK
1. The core of the Transformer's processing lies in the Transformer block, which comprises multi-head self-attention and a Multi-Layer Perceptron layer.
2. Most models consist of multiple such blocks that are stacked sequentially one after the other.
3. The token representations evolve through layers, from the first block to the last one, allowing the model to build up an intricate understanding of each token.
4. This layered approach leads to higher-order representations of the input. The GPT-2 (small) model we are examining consists of `12` such blocks.

### Multi-Head Self-Attention

The self-attention mechanism enables the model to capture relationships among tokens in a sequence, so that each token’s representation is influenced by the others. Multiple attention heads allow the model to consider these relationships from different perspectives; for example, one head may capture short-range syntactic links while another tracks broader semantic context.
#### Step 1: Query, Key, and Value Matrices

![](https://poloclub.github.io/transformer-explainer/article_assets/QKV.png)



Figure 2. Computing Query, Key, and Value matrices from the original embedding.

Each token's embedding vector is transformed into three vectors: Query (Q), Key (K), and Value (V). These vectors are derived by multiplying the input embedding matrix with learned weight matrices for Q, K, and V. Here's a web search analogy to help us build some intuition behind these matrices:

- **Query (Q)** is the search text you type in the search engine bar. This is the token you want to _"find more information about"_.
- **Key (K)** is the title of each web page in the search result window. It represents the possible tokens the query can attend to.
- **Value (V)** is the actual content of web pages shown. Once we matched the appropriate search term (Query) with the relevant results (Key), we want to get the content (Value) of the most relevant pages.

By using these QKV values, the model can calculate attention scores, which determine how much focus each token should receive when generating predictions.

#### Step 2: Multi-Head Splitting

Query, key, and Value vectors are split into multiple heads—in GPT-2 (small)'s case, into `12` heads. Each head processes a segment of the embeddings independently, capturing different syntactic and semantic relationships. This design facilitates parallel learning of diverse linguistic features, enhancing the model's representational power.

#### Step 3: Masked Self-Attention

In each head, we perform masked self-attention calculations. This mechanism allows the model to generate sequences by focusing on relevant parts of the input while preventing access to future tokens.

![](https://poloclub.github.io/transformer-explainer/article_assets/attention.png)

Figure 3. Using Query, Key, and Value matrices to calculate masked self-attention.

- **Dot Product**: The dot product of Query and Key matrices determines the **attention score**, producing a square matrix that reflects the relationship between all input tokens.
- **Scaling · Mask**: The attention scores are scaled and a mask is applied to the upper triangle of the attention matrix to prevent the model from accessing future tokens, setting these values to negative infinity. The model needs to learn how to predict the next token without “peeking” into the future.
- **Softmax · Dropout**: After masking and scaling, the attention scores are converted into probabilities by the softmax operation, then optionally regularized with dropout. Each row of the matrix sums to one and indicates the relevance of every other token to the left of it.

#### Step 4: Output and Concatenation

The model uses the masked self-attention scores and multiplies them with the Value matrix to get the final output of the self-attention mechanism. GPT-2 has `12` self-attention heads, each capturing different relationships between tokens. The outputs of these heads are concatenated and passed through a linear projection.

## ### MLP: Multi-Layer Perceptron

![](https://poloclub.github.io/transformer-explainer/article_assets/mlp.png)

Figure 4. Using MLP layer to project the self-attention representations into higher dimensions to enhance the model's representational capacity.

After the multiple heads of self-attention capture the diverse relationships between the input tokens, the concatenated outputs are passed through the Multilayer Perceptron (MLP) layer to enhance the model's representational capacity.
2.  The MLP block consists of two linear transformations with a [GELU](https://en.wikipedia.org/wiki/Rectified_linear_unit#Gaussian-error_linear_unit_\(GELU\)) activation function in between.
3. The first linear transformation expands the dimensionality of the input four-fold from `768` to `3072`. This expansion step allows the model to project the token representations into a higher-dimensional space, where it can capture richer and more complex patterns that may not be visible in the original dimension.
4. The second linear transformation then reduces the dimensionality back to the original size of `768`.This compression step brings the representations back to a manageable size while retaining the useful nonlinear transformations introduced in the expansion step.
##### Unlike the self-attention mechanism, which integrates information across tokens, the MLP processes tokens independently and simply maps each token representation from one space to another, enriching the overall model capacity.

## Output Probabilities

1. After the input has been processed through all Transformer blocks, the output is passed through the final linear layer to prepare it for token prediction. This layer projects the final representations into a `50,257` dimensional space, where every token in the vocabulary has a corresponding value called `logit`.
2. Any token can be the next word, so this process allows us to simply rank these tokens by their likelihood of being that next word.
3. We then apply the softmax function to convert the logits into a probability distribution that sums to one. This will allow us to sample the next token based on its likelihood.

![](https://poloclub.github.io/transformer-explainer/article_assets/softmax.png)

Figure 5. Each token in the vocabulary is assigned a probability based on the model's output logits. These probabilities determine the likelihood of each token being the next word in the sequence.

The final step is to generate the next token by sampling from this distribution The `temperature` hyperparameter plays a critical role in this process. Mathematically speaking, it is a very simple operation: model output logits are simply divided by the `temperature`:

- `temperature = 1`: Dividing logits by one has no effect on the softmax outputs.
- `temperature < 1`: Lower temperature makes the model more confident and deterministic by sharpening the probability distribution, leading to more predictable outputs.
- `temperature > 1`: Higher temperature creates a softer probability distribution, allowing for more randomness in the generated text – what some refer to as model _“creativity”_.

In addition, the sampling process can be further refined using `top-k` and `top-p` parameters:

- `top-k sampling`: Limits the candidate tokens to the top k tokens with the highest probabilities, filtering out less likely options.
- `top-p sampling`: Considers the smallest set of tokens whose cumulative probability exceeds a threshold p, ensuring that only the most likely tokens contribute while still allowing for diversity.

By tuning `temperature`, `top-k`, and `top-p`, you can balance between deterministic and diverse outputs, tailoring the model's behavior to your specific needs.