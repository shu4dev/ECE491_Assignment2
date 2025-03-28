import torch
from nltk import word_tokenize
from cs336_basics.model import TextClassifier
from huggingface_hub import snapshot_download
model_dir = snapshot_download(repo_id="shu4dev/quality")
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

class quality_classifier:
    def __init__(self):
        self.model_dir = model_dir
        self.model = TextClassifier.from_pretrained(self.model_dir)
        self.context_length = self.model.context_length
        self.model.eval()

    def char_tokenize(self, text: str, max_length: int) -> torch.LongTensor:
        tokens = [ord(c) % 10000 for c in text]
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [0] * (max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)
    
    def predict(self, text: str) -> tuple[int, float]:
        input_ids = self.char_tokenize(text, 512)
        input_ids = input_ids.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(input_ids)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            confidence = probabilities.max().item()
            print("Predicted class:", predicted_class.item())
            print("Confidence score:", confidence)
            return predicted_class.item(), confidence      

if __name__=='__main__':
    model = gopher()
    for i in range(20):
        file_path = f'data/extract_warc{i+1}.txt'
        with open(file_path) as f:
            text = f.read()
            pred = model.classify_quality(text)
            print(f'Sample{i+1}')
            print("-"*20)
            if pred == True:
                print(f'Sample{i+1} is high-quality')
            else:
                print(f'Sample{i+1} is low-quality')
            print("-"*20)