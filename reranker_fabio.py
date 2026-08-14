#=================================
#Author: James Gillespie
#V1.0
#14 Aug '26
#=================================

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- File paths ---
lookup_path = "/Users/jamesgillespie/Documents/FD_just_food.csv"

# --- Load CSVs ---
lookup_df = pd.read_csv(lookup_path)
small_df = pd.read_excel('/Users/jamesgillespie/Downloads/1FoodEx2-to-FABIO Summary Results 18Jun2026.xlsx')
smalldf=small_df[small_df["Step3_Status"]=="ASSIGNED"]

# --- Extract relevant columns ---
lookup_items = lookup_df.iloc[:, 2].astype(str).str.strip()      # adjust column if needed

# --- Create DataFrame from lookup_items ---
df = pd.DataFrame(lookup_items)
print(df)
print(df.head())

df2=smalldf
df2 = df2.reset_index(drop=True)

categories = df["item"].tolist()

terms = df2["L7_desc"][:].tolist()
term4 = df2["L7_desc"][:].tolist()
term3 = df2["L6_desc"][:].tolist()
term2 = df2["L5_desc"][:].tolist()
term1 = df2["L1_desc"][:].tolist()


for i in range(0, len(terms), 2):
    batch_terms = terms[i:i+2]


# ============================================================
# Model & tokenizer
# ============================================================
MODEL_PATH = "/Users/jamesgillespie/Downloads/Qwen3/Qwen3-Reranker-4B"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    padding_side="left"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    attn_implementation="eager"
).to("mps").eval()



# ============================================================
# Tokens and limits
# ============================================================
token_true_id = tokenizer.convert_tokens_to_ids("yes")
token_false_id = tokenizer.convert_tokens_to_ids("no")

max_length = 512


INSTRUCTION = (
    "Given a food or agricultural product term, "
    "judge whether the category correctly describes it. "
    "Answer only yes or no."
)

prefix = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. "
    "Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n"
    "<|im_start|>user\n"
)

suffix = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)

prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

# ============================================================
# Formatting helpers
# ============================================================
def format_pair(term, category):
    return (
        f"<Instruct>: {INSTRUCTION}\n"
        f"<Query>: {term}\n"
        f"<Document>: {category}"
    )

def process_inputs(texts):
    full_texts = [
        prefix + t + suffix
        for t in texts
    ]

    inputs = tokenizer(
        full_texts,
        padding=True,           # dynamic padding
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {k: v.to(model.device) for k, v in inputs.items()}



# ============================================================
# Scoring (YES probability)
# ============================================================
@torch.no_grad()
def compute_yes_scores(inputs):
    logits = model(**inputs).logits[:, -1, :]
    yes_logits = logits[:, token_true_id]
    no_logits = logits[:, token_false_id]

    scores = torch.stack([no_logits, yes_logits], dim=1)
    scores = torch.nn.functional.log_softmax(scores, dim=1)

    return scores[:, 1].exp().tolist()  # P(yes)


def rerank_food_categories(term, categories, batch_size=16, top_k=5):
    scores = {}

    for i in range(0, len(categories), batch_size):
        chunk = categories[i:i + batch_size]

        texts = [format_pair(term, cat) for cat in chunk]
        inputs = process_inputs(texts)

        yes_scores = compute_yes_scores(inputs)

        for cat, score in zip(chunk, yes_scores):
            scores[cat] = float(score)

        del inputs
        torch.mps.empty_cache()

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_category, best_score = ranked[0]
    top_categories = ranked[:top_k]

    return best_category, best_score, scores, top_categories


def build_ontology_string_dedup(terms):
    """
    terms: list ordered term1 → term7 (general → specific)

    Returns:
        ontology string with same-granularity terms omitted
    """
    cleaned = [t for t in terms if isinstance(t, str) and t.strip()]

    deduped = []
    last_norm = None

    for t in cleaned:
        norm = t
        if norm != last_norm:
            deduped.append(t.strip())
            last_norm = norm


    return " belonging to ".join(deduped)


df2["best_category"] = None
df2["best_score"] = np.nan
df2["reranker_rank"] = np.nan
df2["reranker_score"] = np.nan

for i in range(0,4181):

    finalT = build_ontology_string_dedup(
        [term4[i], term3[i], term2[i], term1[i]]
    )

    term = finalT

    best_category, best_score, scores, top3 = rerank_food_categories(
        term, categories, 32
    )

    # Best reranker result
    df2.loc[i, "best_category"] = best_category
    df2.loc[i, "best_score"] = best_score

    # Step 3 expert/reference category
    expert_category = df2.loc[i, "Step3_Final_FABIO"]

    # Rank all categories according to reranker score
    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Find the expert category in the reranker results
    for rank, (category, score) in enumerate(ranked, start=1):
        if category == expert_category:
            df2.loc[i, "reranker_rank"] = rank
            df2.loc[i, "reranker_score"] = score
            break

    print(f"{i}/{len(terms)} processed")