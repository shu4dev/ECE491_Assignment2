import nltk
from nltk import word_tokenize
nltk.download('punkt_tab')
class gopher:
    def classify_quality(self, text: str) -> bool:
        tokens = word_tokenize(text)
        if len(tokens) < 50 or len(tokens) > 100000:
            return False
        mean = sum(len(token) for token in tokens) / len(tokens)
        if mean < 3 or mean > 10:
            return False
        lines = text.split("\n")
        ellipsis_lines = [line for line in lines if line.endswith("...")]
        if len(ellipsis_lines) / len(lines) > 0.3:
            return False
        alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
        if len(alpha_tokens) / len(tokens) < 0.8:
            return False
        return True