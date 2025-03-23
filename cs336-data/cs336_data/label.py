import os
import glob
import csv
import re
from typing import List
from huggingface_hub import hf_hub_download
import fasttext

model = fasttext.load_model(
    hf_hub_download("kenhktsui/llm-data-textbook-quality-fasttext-classifer-v1", "model.bin")
)

def replace_newlines(text: str) -> str:
    return re.sub("\n+", " ", text)

def predict(text_list: List[str]) -> List[dict]:
    text_list = [replace_newlines(text) for text in text_list]
    pred = model.predict(text_list)
    return [{"label": l[0].lstrip("__label__"), "score": s[0]} for l, s in zip(*pred)]


folder_path = 'data/positive_sample'
file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
output_rows = []

for file_path in file_paths:
    with open(file_path, 'r') as file:
        text = file.read()
    prediction = predict([text])[0]
    output_rows.append((file_path, prediction["label"]))

csv_file = 'data/positive_sample/predictions.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["file_path", "label"])
    writer.writerows(output_rows)