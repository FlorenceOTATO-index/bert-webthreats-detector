"""
WebShell Detector using BERT + LSTM
-----------------------------------

This script implements a WebShell detection pipeline using BERT embeddings and an LSTM classifier.
It loads and preprocesses WebShell datasets, decodes Base64-encoded payloads, and trains a
classification model to distinguish between benign and malicious (WebShell) samples.

Key Features:
- Loads multiple WebShell datasets (`webshell.csv` and `webshell_data.csv`)
- Cleans and decodes Base64-encoded content into opcode-like sequences
- Uses BERT for token embeddings combined with an LSTM for sequential feature extraction
- Handles class imbalance with weighted sampling
- Trains, validates, and evaluates the model with Accuracy, F1, Recall, and Precision metrics
- Saves the best-performing model and supports inference on new samples

Output:
- Trained model (`best_model.pth`)
- Console logs with training progress, evaluation metrics, and sample predictions
"""

import warnings
import time
import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import base64
import re
from collections import Counter


# 1. Load datasets
column_names = ['content', 'label']
df1 = pd.read_csv('/Users/florence/Desktop/webshell.csv',
                  header=None, names=column_names)
df2 = pd.read_csv('/Users/florence/Desktop/webshell_data.csv',
                  header=None, names=column_names)
df1['label'] = df1['label'].fillna(0).astype(int)
df2['label'] = df2['label'].fillna(0).astype(int)
print(f"webshell.csv dataset size: {df1.shape}")
print(f"webshell_data.csv dataset size: {df2.shape}")
df = pd.concat([df1, df2], axis=0, ignore_index=True)
print(f"Combined dataset size: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")


# 2. Configuration class
class Config:
    def __init__(self):
        self.max_len = 256
        self.batch_size = 32
        self.epochs = 1
        self.learning_rate = 2e-5
        self.hidden_size = 128
        self.bert_path = "/Users/florence/Desktop/bert-base-cased"  # Local model path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = 42


config = Config()
config.bert_path = "/Users/florence/Desktop/bert-base-cased"

# Load tokenizer and model from local path
tokenizer = BertTokenizer.from_pretrained(
    config.bert_path,
    local_files_only=True
)
model = BertModel.from_pretrained(
    config.bert_path,
    local_files_only=True
)


# 3. WebShell Detector class
class WebShellDetector:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(
            config.bert_path,
            local_files_only=True
        )
        self.model = BERTLSTM(config)
        self.model.to(config.device)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.config.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def predict(self, base64_content):
        try:
            start_time = time.time()
            decoded = base64.b64decode(base64_content).decode('utf-8', errors='ignore')
            cleaned = re.sub(r'[^A-Z0-9_ ]', ' ', decoded.upper())

            encoding = self.tokenizer.encode_plus(
                cleaned,
                add_special_tokens=True,
                max_length=self.config.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            with torch.no_grad():
                outputs = self.model(
                    encoding['input_ids'].to(self.config.device),
                    encoding['attention_mask'].to(self.config.device)
                )
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs).item()
            inference_time = (time.time() - start_time) * 1000

            return {
                'prediction': pred,
                'confidence': probs[0][pred].item(),
                'opcode': cleaned[:200] + '...' if len(cleaned) > 200 else cleaned
            }
        except Exception as e:
            return {'error': str(e)}


# 4. Helper function: clean Base64 strings
def clean_base64(content):
    """Clean Base64 string by removing whitespace and fixing padding"""
    cleaned = re.sub(r'\s+', '', content)
    if len(cleaned) % 4 != 0:
        cleaned += '=' * (4 - len(cleaned) % 4)
    return cleaned


# 5. Data preprocessing
def preprocess_data(df):
    opcode_sequences = []
    labels = []
    skip_count = 0

    for _, row in df.iterrows():
        try:
            content = str(row['content']).strip()
            if not content:
                skip_count += 1
                continue

            cleaned_base64 = clean_base64(content)

            try:
                decoded = base64.b64decode(cleaned_base64)

                try:
                    text = decoded.decode('utf-8')
                except UnicodeDecodeError:
                    text = str(decoded)

                text = re.sub(r'[^\w\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip().upper()

                if not text:
                    skip_count += 1
                    continue

                try:
                    label = int(float(row['label']))
                    if label not in (0, 1):
                        skip_count += 1
                        continue
                except:
                    skip_count += 1
                    continue

                opcode_sequences.append(text)
                labels.append(label)

            except base64.binascii.Error as e:
                print(f"Base64 decode failed (first 30 chars: {content[:30]}): {str(e)}")
                skip_count += 1

        except Exception as e:
            warnings.warn(f"Error processing row: {str(e)}")
            skip_count += 1

    print(f"\nPreprocessing summary:")
    print(f"- Successful: {len(opcode_sequences)}")
    print(f"- Skipped: {skip_count}")
    print(f"- Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return opcode_sequences, labels


# 6. DataLoader creation
def create_data_loaders(opcode_sequences, labels, tokenizer, config):
    train_opcodes, val_opcodes, train_labels, val_labels = train_test_split(
        opcode_sequences, labels, test_size=0.2,
        random_state=config.random_state, stratify=labels
    )

    class_counts = torch.tensor([(torch.tensor(train_labels) == 0).sum().item(),
                                 (torch.tensor(train_labels) == 1).sum().item()])
    class_weights = 1. / class_counts
    sample_weights = class_weights[torch.tensor(train_labels)]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataset = WebShellDataset(train_opcodes, train_labels, tokenizer, config.max_len)
    val_dataset = WebShellDataset(val_opcodes, val_labels, tokenizer, config.max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


# 7. Dataset class
class WebShellDataset(Dataset):
    def __init__(self, opcode_sequences, labels, tokenizer, max_len):
        self.opcode_sequences = opcode_sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.opcode_sequences)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.opcode_sequences[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# 8. Model class: BERT + LSTM
class BERTLSTM(nn.Module):
    def __init__(self, config):
        super(BERTLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=config.hidden_size,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)
        output = self.dropout(lstm_output[:, -1, :])
        return self.classifier(output)


# 9. Training and evaluation
def train_model(model, train_loader, val_loader, optimizer, criterion, config):
    model.to(config.device)
    best_val_acc = 0.0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(
                batch['input_ids'].to(config.device),
                batch['attention_mask'].to(config.device)
            )
            loss = criterion(outputs, batch['label'].to(config.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{config.epochs} | Batch {batch_idx}/{len(train_loader)}"
                      f" | Loss: {loss.item():.4f}")

        val_acc, val_f1, val_recall, val_precision = evaluate_model(model, val_loader, config)
        print(f'Epoch {epoch + 1}/{config.epochs} | Loss: {total_loss / len(train_loader):.4f} | '
              f'Val Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Recall: {val_recall:.4f} | Precision: {val_precision:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'Training complete. Best Val Acc: {best_val_acc:.4f}')


def evaluate_model(model, data_loader, config):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(
                batch['input_ids'].to(config.device),
                batch['attention_mask'].to(config.device)
            )
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['label'].cpu().numpy())

    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)
    return (
        accuracy_score(true_labels, predictions),
        f1_score(true_labels, predictions),
        recall_score(true_labels, predictions),
        precision_score(true_labels, predictions)
    )


# 10. Main
def main():
    opcode_sequences, labels = preprocess_data(df)
    print(f"\nValid samples: {len(opcode_sequences)}")
    print(f"Label distribution: {Counter(labels)}")

    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    train_loader, val_loader = create_data_loaders(opcode_sequences, labels, tokenizer, config)

    model = BERTLSTM(config)
    class_weights = torch.tensor([1.0, len(labels) / sum(labels)]).to(config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_model(model, train_loader, val_loader, optimizer, criterion, config)

    detector = WebShellDetector(config)
    detector.load_model('best_model.pth')

    test_samples = [
        df.iloc[0]['content'],
        df.iloc[-1]['content'],
        "U0hFTEwgWkVORF9FVkFMIEZJTEVfR0VU"
    ]

    print("\n=== Prediction Tests ===")
    for i, sample in enumerate(test_samples):
        result = detector.predict(sample)
        print(f"\nSample {i + 1}:")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Prediction: {'WebShell' if result['prediction'] == 1 else 'Normal'}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Opcode preview: {result['opcode']}")


if __name__ == '__main__':
    main()
