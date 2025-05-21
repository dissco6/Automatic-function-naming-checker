import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re

def prepare_input(name, original_code):
    modified_code = re.sub(r'def\s+\w+', 'def FUNCNAME', original_code, count=1)
    return f"{name}\n{modified_code}"

model = RobertaForSequenceClassification.from_pretrained("./final_codebert_model_3", num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

samples = [
    ("recommend_friends_by_mutuals", """def recommend_friends_by_mutuals(user_id: int, friendships: dict[int, set[int]]) -> list[int]:
    user_friends = friendships.get(user_id, set())
    recommendations = {}
    for friend in user_friends:
        for mutual in friendships.get(friend, set()):
            if mutual != user_id and mutual not in user_friends:
                recommendations[mutual] = recommendations.get(mutual, 0) + 1
    return sorted(recommendations, key=recommendations.get, reverse=True)

    """),
    ("new_f_name", """def recommend_friends_by_mutuals(user_id: int, friendships: dict[int, set[int]]) -> list[int]:
        user_friends = friendships.get(user_id, set())
        recommendations = {}
        for friend in user_friends:
            for mutual in friendships.get(friend, set()):
                if mutual != user_id and mutual not in user_friends:
                    recommendations[mutual] = recommendations.get(mutual, 0) + 1
        return sorted(recommendations, key=recommendations.get, reverse=True)
    """),
]

for name_to_test, original_code in samples:
    input_text = prepare_input(name_to_test, original_code)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() - 0.0

    result = "good" if pred == 1 else "bad"
    print(f"Funkcija`{name_to_test}`: {result}")
