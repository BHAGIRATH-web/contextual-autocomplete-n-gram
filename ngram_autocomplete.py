import re
from collections import defaultdict, Counter

# ---------------- Load Dataset From File ----------------
def load_dataset(path="data/data.txt"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print("⚠ Dataset file not found at:", path)
        return ""

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
                    candidates[word] += count / self.context_counts[k][context]
            if found:
                break
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]

# ---------------- Main ----------------
if __name__ == "__main__":
    print("🔄 Loading dataset and training model...")
    model = NGramModel(n=3)
    training_text = load_dataset("data/data.txt")
    if not training_text:
        print("No dataset found — falling back to small sample for demo.")
        training_text = """
        artificial intelligence is transforming the world.
        next word prediction helps typing faster.
        language models learn patterns from text.
        a simple n gram model predicts next words.
        """
    model.train(training_text)
    print("✅ Training complete.\n--- N-Gram Autocomplete Demo ---")

    while True:
        user_input = input("\nEnter phrase (or 'exit'): ")
        if user_input.strip().lower() == "exit":
            break
        preds = model.predict(user_input, top_k=6)
        if not preds:
            print("  (no suggestions)")
        else:
            print("Suggestions:")
            for w, s in preds:
                print(f"  👉 {w}  ({s:.2f})")
