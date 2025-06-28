from datasets import load_dataset

# Load dataset from HF hub

# (anchor, positive, negative)
all_nli_triplet_train = load_dataset("sentence-transformers/all-nli", "triplet", split="train[:500]", cache_dir="./dataset")
# (sentence1, sentence2) + score
stsb_pair_score_train = load_dataset("sentence-transformers/stsb", split="train[:500]", cache_dir="./dataset")

# (anchor, positive, negative)
all_nli_triplet_dev = load_dataset("sentence-transformers/all-nli", "triplet", split="dev[:400]", cache_dir="./dataset")
# (sentence1, sentence2, score)
stsb_pair_score_dev = load_dataset("sentence-transformers/stsb", split="validation[:400]", cache_dir="./dataset")