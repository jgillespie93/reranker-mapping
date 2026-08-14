# Qwen 3 Reranker for Food Category Matching

This repository contains the Qwen 3 Reranker implementation used in **Step 4 of the automated food classification pipeline** described in the associated paper.

The model evaluates a food or agricultural product descriptor against candidate ontology categories and assigns each candidate a probability of being a valid match.

## Method

The reranker is used as a **cross-encoder**. The product descriptor and candidate category are provided together to the model, which evaluates whether the category correctly describes the product.

The instruction used was:

```python
INSTRUCTION = (
    "Given a food or agricultural product term, "
    "judge whether the category correctly describes it. "
    "Answer only yes or no."
)
```

The system prompt constrains the model output to `yes` or `no`.

Each candidate is assigned a score based on the model's probability of `yes`:

[
P(\mathrm{yes}) =
\frac{\exp(L_{\mathrm{yes}})}
{\exp(L_{\mathrm{yes}})+\exp(L_{\mathrm{no}})}
]

where (L_{\mathrm{yes}}) and (L_{\mathrm{no}}) are the logits corresponding to the `yes` and `no` tokens.

Candidates are then ranked according to (P(\mathrm{yes})), with the highest-scoring category selected as the automated assignment.

## Input

The query consists of the hierarchical food descriptor constructed from the ontology terms:

```text
term4 belonging to term3 belonging to term2 belonging to term1
```

For example:

```text
<product descriptor>
    ↓
Qwen 3 Reranker
    ↓
Candidate categories
    ↓
P(yes) for each category
    ↓
Ranked categories
```

The implementation evaluates candidates in batches to reduce memory requirements.

## Configuration

| Parameter               | Value                               |
| ----------------------- | ----------------------------------- |
| Model                   | Qwen 3 Reranker                     |
| Task                    | Food/agricultural category matching |
| Output                  | Probability of `yes`                |
| Maximum sequence length | 512 tokens                          |
| Batch size              | 32 in the reported implementation   |
| Ranking                 | Descending (P(\mathrm{yes}))        |
| Output                  | Best category + top-k candidates    |

The implementation retains the **top 5 candidate categories** for subsequent analysis.

## Implementation

The core scoring operation is:

```python
@torch.no_grad()
def compute_yes_scores(inputs):
    logits = model(**inputs).logits[:, -1, :]

    yes_logits = logits[:, token_true_id]
    no_logits = logits[:, token_false_id]

    scores = torch.stack([no_logits, yes_logits], dim=1)
    scores = torch.nn.functional.log_softmax(scores, dim=1)

    return scores[:, 1].exp().tolist()
```

Candidates are subsequently ranked by their `yes` probability:

```python
ranked = sorted(
    scores.items(),
    key=lambda x: x[1],
    reverse=True
)
```

## Important distinction

The Qwen 3 Reranker **does not generate standalone sentence embeddings in this implementation**. It functions as a cross-encoder that jointly processes the descriptor and candidate category and produces a relevance score.

This allows it to consider interactions between the two texts that are not captured by independently generated embeddings, making it suitable for fine-grained reranking of semantically similar food and agricultural categories.

## Reproducibility

The following should be recorded alongside the analysis:

* Exact Qwen 3 Reranker model identifier
* Transformers version
* PyTorch version
* Tokenizer configuration
* Maximum sequence length (512)
* Candidate category list
* Batch size
* Number of candidates retained (`top_k`)
* Hardware used for inference

The reranker outputs should be considered **automated classification recommendations** rather than definitive assignments, as fine-grained food classification can require domain-specific knowledge and expert judgement.

