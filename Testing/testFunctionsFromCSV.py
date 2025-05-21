import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import re

def prepare_input(name, original_code):
    """Replaces function name in code with 'FUNCNAME' and prepends the actual name."""
    modified_code = re.sub(r'def\s+\w+', 'def FUNCNAME', original_code, count=1)
    return f"{name}\n{modified_code}"

model = RobertaForSequenceClassification.from_pretrained("./final_codebert_model_3", num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model.eval()

df = pd.read_csv("open_source_func_filtered.csv", sep=";")
df = df.dropna(subset=["name", "code"])  

total = len(df)
good_predictions = 0
bad_predictions = 0
bad_names = []

for idx, row in df.iterrows():
    input_text = prepare_input(row['name'], row['code'])
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    if pred == 1:
        good_predictions += 1
    else:
        bad_predictions += 1
        bad_names.append(row['name'])

print(f"Total functions evaluated: {total}")
print(f"Good: {good_predictions}")
print(f"bad: {bad_predictions}")
true_positive_rate = good_predictions / total if total > 0 else 0
print(f"TPR: {true_positive_rate:.2%}")

with open("bad_function_names_from_org.txt", "w", encoding="utf-8") as f:
    for name in bad_names:
        f.write(f"{name}\n")
