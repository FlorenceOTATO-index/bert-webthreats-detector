import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from collections import Counter
import json
import base64

# System optimization
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())

# Device configuration
device = torch.device("cpu")


def preprocess_text(text):
    """Handle mixed format data (base64 and raw text)"""
    try:
        if len(text) > 20 and ' ' not in text and text.isprintable():
            decoded = base64.b64decode(text).decode('utf-8', errors='ignore')
            return decoded if len(decoded) > 0 else text
        return text
    except:
        return text


class MultiTaskBERT(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model
        hidden_size = base_model.config.hidden_size

        # Shared layers
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # Task-specific heads
        self.xss_head = nn.Linear(256, 2)  # For labels 0/1
        self.webshell_head = nn.Linear(256, 2)  # For labels 2/3 (mapped to 0/1)
        self.task_router = nn.Linear(hidden_size, 2)  # Decide which head to use

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        features = self.shared_layer(pooled)

        # Get task probabilities
        task_probs = torch.softmax(self.task_router(pooled), dim=1)

        return {
            'task_probs': task_probs,
            'xss_logits': self.xss_head(features),
            'ws_logits': self.webshell_head(features)
        }


class HybridDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        # 将pandas Series转换为numpy数组或list
        self.texts = [preprocess_text(str(t)) for t in texts.values]  # 使用.values获取数组
        self.labels = labels.values  # 转换为numpy数组
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 确保idx在合理范围内
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with size {len(self)}")

        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Create task indicator (0 for XSS, 1 for WebShell)
        label = self.labels[idx]
        task_indicator = 0 if label in [0, 1] else 1

        # Map labels to 0/1 for each task
        xss_label = label if task_indicator == 0 else 0
        ws_label = (label - 2) if task_indicator == 1 else 0

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor([task_indicator, xss_label, ws_label], dtype=torch.long)
        }

def load_data():
    """Load data with 4-class labels:
    0: xss benign, 1: xss malicious
    2: webshell benign, 3: webshell malicious
    """
    # Load XSS data (first 1000 rows)
    xss_data = pd.read_csv('XSS_dataset.csv', header=None)
    xss_data = xss_data.iloc[:, [1, 2]]
    xss_data.columns = ['text', 'is_xss']
    xss_data['label'] = xss_data['is_xss'].map({0: 0, 1: 1})  # 0=benign, 1=malicious

    # Load WebShell data
    def load_ws_data(path, is_malicious):
        df = pd.read_csv(path, header=None)
        df = df.iloc[:, [0]]
        df.columns = ['text']
        df['label'] = 2 if not is_malicious else 3  # 2=benign, 3=malicious
        return df

    df = pd.concat([
        xss_data[['text', 'label']],
        load_ws_data('../webshell/benign.csv', False),
        load_ws_data('../webshell/malicious.csv', True)
    ])

    print("\n=== Label Distribution ===")
    print(df['label'].value_counts())
    print("0: xss_benign | 1: xss_malicious | 2: ws_benign | 3: ws_malicious")

    return df


def evaluate(model, dataloader, task_name):
    """Evaluate and print classification report for a specific task"""
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            labels = batch['labels'].to(device)
            outputs = model(**inputs)

            # Get predictions based on task
            if task_name == 'xss':
                mask = (labels[:, 0] == 0)  # XSS samples
                if mask.any():
                    logits = outputs['xss_logits'][mask]
                    preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    true_labels.extend(labels[mask, 1].cpu().numpy())
            else:  # webshell
                mask = (labels[:, 0] == 1)  # WebShell samples
                if mask.any():
                    logits = outputs['ws_logits'][mask]
                    preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    true_labels.extend(labels[mask, 2].cpu().numpy())

    if len(true_labels) > 0:
        print(f"\n=== {task_name.upper()} Evaluation ===")
        report = classification_report(
            true_labels, preds,
            target_names=['benign', 'malicious'],
            digits=4,
            output_dict=True
        )
        print(classification_report(
            true_labels, preds,
            target_names=['benign', 'malicious'],
            digits=4
        ))
        return report


def train():
    df = load_data()
    model_save_path = "../codebert/small-v2/xss+webshellModel.pt"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Split into XSS and WebShell datasets
    xss_data = df[df['label'].isin([0, 1])]
    ws_data = df[df['label'].isin([2, 3])]

    # Split XSS data
    xss_train, xss_test = train_test_split(
        xss_data, test_size=0.2, stratify=xss_data['label'], random_state=42
    )

    # Split WebShell data
    ws_train, ws_test = train_test_split(
        ws_data, test_size=0.2, stratify=ws_data['label'], random_state=42
    )

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../codebert/small-v2")
    model = MultiTaskBERT(AutoModel.from_pretrained("../codebert/small-v2")).to(device)

    # Create data loaders
    train_texts = pd.concat([xss_train, ws_train])['text'].reset_index(drop=True)
    train_labels = pd.concat([xss_train, ws_train])['label'].reset_index(drop=True)

    train_loader = DataLoader(
        HybridDataset(train_texts, train_labels, tokenizer),
        batch_size=32,
        shuffle=True
    )

    xss_test_loader = DataLoader(
        HybridDataset(xss_test['text'], xss_test['label'], tokenizer),
        batch_size=32, shuffle=False
    )

    ws_test_loader = DataLoader(
        HybridDataset(ws_test['text'], ws_test['label'], tokenizer),
        batch_size=32, shuffle=False
    )

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Load previous model if exists
    prev_model = None
    if os.path.exists(model_save_path):
        prev_model = MultiTaskBERT(AutoModel.from_pretrained("../codebert/small-v2")).to(device)
        prev_model.load_state_dict(torch.load(model_save_path))
        prev_model.eval()

    # Training loop
    for epoch in range(2):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)

            # Calculate loss for active tasks
            task_mask = (labels[:, 0] == 0)
            xss_loss = criterion(outputs['xss_logits'][task_mask], labels[task_mask, 1]) if task_mask.any() else 0

            task_mask = (labels[:, 0] == 1)
            ws_loss = criterion(outputs['ws_logits'][task_mask], labels[task_mask, 2]) if task_mask.any() else 0

            loss = xss_loss + ws_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"\nEpoch {epoch + 1} Loss: {epoch_loss / len(train_loader):.4f}")

    # Evaluate final model
    print("\n=== Evaluating Final Model ===")
    final_xss_report = evaluate(model, xss_test_loader, 'xss')
    final_ws_report = evaluate(model, ws_test_loader, 'webshell')

    final_xss_f1 = final_xss_report['weighted avg']['f1-score'] if final_xss_report else 0
    final_ws_f1 = final_ws_report['weighted avg']['f1-score'] if final_ws_report else 0
    final_avg_f1 = (final_xss_f1 + final_ws_f1) / 2

    # Evaluate previous model if exists
    prev_avg_f1 = 0
    if prev_model is not None:
        print("\n=== Evaluating Previous Model ===")
        prev_xss_report = evaluate(prev_model, xss_test_loader, 'xss')
        prev_ws_report = evaluate(prev_model, ws_test_loader, 'webshell')

        prev_xss_f1 = prev_xss_report['weighted avg']['f1-score'] if prev_xss_report else 0
        prev_ws_f1 = prev_ws_report['weighted avg']['f1-score'] if prev_ws_report else 0
        prev_avg_f1 = (prev_xss_f1 + prev_ws_f1) / 2

    # Model comparison and saving logic
    if prev_model is None or final_avg_f1 > prev_avg_f1:
        print(f"\nNew model {'created' if prev_model is None else 'improved'}. Saving to {model_save_path}")
        torch.save(model.state_dict(), model_save_path)
        best_model = model
    else:
        print("\nNew model did not improve. Keeping previous model.")
        best_model = prev_model

    # Final evaluation with best model
    print("\n=== Best Model Evaluation ===")
    evaluate(best_model, xss_test_loader, 'xss')
    evaluate(best_model, ws_test_loader, 'webshell')

    return best_model


if __name__ == "__main__":
    best_model = train()
