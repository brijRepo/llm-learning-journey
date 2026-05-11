from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_sim(a: str, b: str) -> float:
    embs = model.encode([a, b])
    dot = np.dot(embs[0], embs[1])
    return float(dot / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1])))


def euclidean_dist(a: str, b: str) -> float:
    embs = model.encode([a, b])
    return float(np.linalg.norm(embs[0] - embs[1]))


print("=== EXPERIMENT 1: Paraphrase Detection ===\n")

pairs = [
    ("How does attention work in transformers?",
     "Explain the attention mechanism in transformer architecture."),
    
    ("What is RAG?",
     "Define Retrieval Augmented Generation."),

    ("What is RAG?",
     "What is the weather in Mumbai today?"),

    ("I love machine learning.",
     "I hate machine learning."),

    ("The model is fast.",
     "The model is slow."),
]

for sent_a, sent_b in pairs:
    sim = cosine_sim(sent_a, sent_b)
    print(f"  A: {sent_a}")
    print(f"  B: {sent_b}")
    print(f"  Cosine Similarity: {sim:.4f}")
    print()


print("=== EXPERIMENT 2: Cosine vs Euclidean ===\n")

short = "RAG uses vector search."
long  = """RAG, which stands for Retrieval Augmented Generation,
is a technique in LLM engineering where relevant documents
are retrieved from a vector database using semantic similarity
search, and then injected into the LLM's context window as
additional grounding information before the model generates
its final answer. This dramatically reduces hallucination
because the model is anchored to real retrieved content."""
unrelated = "Cricket is popular in India."

pairs_exp2 = [
    ("Short vs Long (same topic)", short, long),
    ("Short vs Unrelated",         short, unrelated),
    ("Long vs Unrelated",          long,  unrelated),
]

print(f"{'Comparison':<35} {'Cosine Sim':>12} {'Euclidean Dist':>16}")
print("-" * 65)
for label, a, b in pairs_exp2:
    cos = cosine_sim(a, b)
    euc = euclidean_dist(a, b)
    print(f"{label:<35} {cos:>12.4f} {euc:>16.4f}")

print("""
Key Insight:
  Cosine similarity: SHORT and LONG texts on same topic → HIGH (correct)
  Euclidean distance: SHORT and LONG texts on same topic → LARGE (wrong!)
  → Cosine measures DIRECTION (meaning), not MAGNITUDE (length)
""")

print("=== EXPERIMENT 3: Relevance Thresholds ===\n")

query = "Explain the attention mechanism"

candidates = [
    "Multi-head attention computes Q, K, V matrices in parallel.",         # very relevant
    "Self-attention allows tokens to attend to all other tokens.",          # very relevant
    "Transformers were introduced in the 'Attention is All You Need' paper.", # relevant
    "LLMs are trained on large datasets using next-token prediction.",      # somewhat related
    "Python is a popular programming language for data science.",           # weakly related
    "The IPL cricket season starts in April.",                              # irrelevant
]

query_emb = model.encode(query)
cand_embs = model.encode(candidates)

print(f"Query: '{query}'\n")
print(f"{'Score':>7}  {'Label':<14}  Text")
print("-" * 80)


for emb, text in zip(cand_embs, candidates):
    dot = np.dot(query_emb, emb)
    sim = float(dot / (np.linalg.norm(query_emb) * np.linalg.norm(emb)))

    if sim >= 0.60:
        label = "RELEVANT"
    elif sim >= 0.35:
        label = "MARGINAL"
    else:
        label = "IRRELEVANT"

    print(f"{sim:>7.4f}  {label:<14}  {text[:60]}...")

print("""
Production Rule of Thumb:
  > 0.70 → Highly relevant, always include
  0.50–0.70 → Relevant, include if top-k slots available
  0.35–0.50 → Marginal, include only if desperate
  < 0.35 → Irrelevant, EXCLUDE — hurts more than helps
""")