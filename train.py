import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
DATA_PATH = "/content/Reviews.csv"
MODEL_SAVE_PATH = "./fine_tuned_bert_model"
BATCH_SIZE = 16 # Adjust batch size based on your GPU memory
EPOCHS = 2 # Keep small for demonstration; increase for better performance
LEARNING_RATE = 2e-5
MAX_SEQUENCE_LENGTH = 128

# --- Preprocessing Functions ---
def clean_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.split()
    return " ".join(text)

# --- Dataset Class ---
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Main Training Logic ---
if __name__ == '__main__':
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH, on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("Please ensure 'Reviews.csv' is uploaded to /content/.")
        exit()

    print("Cleaning text...")
    df['clean_review'] = df['Text'].apply(clean_text)
    df['sentiment_label'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)

    # Filter out rows where 'clean_review' might be empty after cleaning
    df = df[df['clean_review'].str.strip() != '']
    if df.empty:
        print("No valid reviews left after cleaning. Exiting.")
        exit()

    print("Initializing tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    dataset = ReviewDataset(
        texts=df['clean_review'].to_numpy(),
        labels=df['sentiment_label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_SEQUENCE_LENGTH
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Starting training on {device} for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0: # Print loss every 100 batches
                print(f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    print("Training complete.")

    # Save the trained model
    print(f"Saving trained model to {MODEL_SAVE_PATH}...")
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("Model and tokenizer saved.")
