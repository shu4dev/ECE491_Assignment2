import torch
from cs336_basics.model import TextClassifier

def char_tokenize(text: str, max_length: int) -> torch.LongTensor:
    # Convert each character to a token using the same encoding as in training
    tokens = [ord(c) % 10000 for c in text]
    # Truncate or pad the token list to match max_length
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens = tokens + [0] * (max_length - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

# Path to your pretrained model directory (should contain model_config.json and model.pt)
model_dir = "/home/shu4/koa_scratch/ECE491_Assignment2/cs336-data/cs336_data/model"
model = TextClassifier.from_pretrained(model_dir)
model.eval()  # Set model to evaluation mode

# Your input string for inference
input_text = "This is a test input."

# Tokenize the string using the character-level approach
input_ids = char_tokenize(input_text, max_length=model.context_length)
# Add batch dimension (batch_size = 1)
input_ids = input_ids.unsqueeze(0)

# Run inference
with torch.no_grad():
    logits = model(input_ids)
    # Convert logits to probabilities using softmax
    probabilities = torch.softmax(logits, dim=-1)
    # Get the predicted class (the index with the highest probability)
    predicted_class = torch.argmax(probabilities, dim=-1)
    # Get the confidence score of the prediction
    confidence = probabilities.max().item()

print("Predicted class:", predicted_class.item())
print("Confidence score:", confidence)