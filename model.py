import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

def clean_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.split()
    return " ".join(text)

# Instantiate tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode_text(text):
    return tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

# Instantiate model (load pre-trained weights, for inference)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
# IMPORTANT: For a truly deployed model, you would save the trained model weights
# after training in the notebook (e.g., model.save_pretrained('./my_model_dir'))
# and then load those specific weights here (e.g., model.from_pretrained('./my_model_dir')).
# Currently, this loads only the base pre-trained BERT weights, not your fine-tuned model.

def predict_email_sentiment(text):
    clean_input = clean_text(text)
    inputs = encode_text(clean_input)

    # Ensure model is in evaluation mode
    model.eval()
    with torch.no_grad():
        output = model(**inputs)

    probs = torch.softmax(output.logits, dim=1)
    fake_prob = probs[0][1].item()

    if fake_prob > 0.5:
        return f"Fake Review ({fake_prob:.2f})"
    else:
        return f"Genuine Review ({1-fake_prob:.2f})"

# Example of how to use the function (for testing within the file if needed)
if __name__ == '__main__':
    # This part will only run if model.py is executed directly
    # It won't run when imported by app.py
    test_review = "This product is absolutely terrible and a complete scam!"
    prediction = predict_email_sentiment(test_review)
    print(f"Test review: '{test_review}' -> Prediction: {prediction}")

    test_review_2 = "This is an amazing product, I love it so much."
    prediction_2 = predict_email_sentiment(test_review_2)
    print(f"Test review: '{test_review_2}' -> Prediction: {prediction_2}")
