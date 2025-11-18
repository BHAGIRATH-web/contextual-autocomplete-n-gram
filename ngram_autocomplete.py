import re
from collections import defaultdict, Counter

# ---------------- Tokenizer ----------------
def tokenize(text):
    text = text.lower()
    text = re.sub(r"([.,!?;:()])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

# ---------------- N-gram Model ----------------
class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(Counter)
        self.vocab = set()

    def train(self, text):
        tokens = tokenize(text)
        padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        self.vocab.update(padded)

        for i in range(len(padded)):
            for k in range(1, self.n + 1):
                if i - k + 1 < 0:
                    continue
                gram = tuple(padded[i - k + 1 : i + 1])
                context = gram[:-1]
                word = gram[-1]

                self.ngram_counts[k][gram] += 1
                self.context_counts[k][context] += 1

    def predict(self, context_words, top_k=5):
        context_words = tokenize(context_words)

        if len(context_words) < self.n - 1:
            context_words = ["<s>"] * (self.n - 1 - len(context_words)) + context_words
        else:
            context_words = context_words[-(self.n - 1):]

        candidates = defaultdict(float)

        # Try highest n-gram to 1-gram (backoff)
        for k in range(self.n, 0, -1):
            context = tuple(context_words[-(k - 1):]) if k > 1 else tuple()

            found = False
            for gram, count in self.ngram_counts[k].items():
                if gram[:-1] == context:
                    found = True
                    word = gram[-1]
                    # Simple probability
                    candidates[word] += count / self.context_counts[k][context]

            if found:
                break  # higher-order match mil gaya

        # Sort predictions
        preds = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return preds[:top_k]


# ---------------- Demo ----------------
if __name__ == "__main__":
    sample_text = """
    artificial intelligence is transforming the world.
    next word prediction helps typing faster.
    language models learn patterns from text.
    a simple n gram model predicts next words.
    """

    model = NGramModel(n=3)
    model.train(sample_text)

    print("\n--- N-Gram Autocomplete Demo ---")
    while True:
        user_input = input("\nEnter a phrase (or 'exit'): ")
        if user_input.lower() == "exit":
            break

        predictions = model.predict(user_input)
        print("\nPredictions:")
        for word, score in predictions:
            print(f"  {word} ({score:.2f})")
