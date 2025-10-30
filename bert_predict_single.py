"""
WebShell File Scanner (BERT + LSTM)
-----------------------------------

This script scans `.php` files within a given folder to detect potential WebShells using a 
BERT + LSTM classifier. It works by checking whether file contents are Base64-encoded, 
decoding them, preprocessing the decoded text, and then predicting with a pre-trained model.

Key Features:
- Detects whether a file is Base64-encoded before processing
- Decodes Base64 safely with cleanup and validation
- Uses BERT embeddings and an LSTM classifier for sequence classification
- Processes multiple files in a directory recursively
- Reports number of scanned files, skipped files, and detected WebShells

Output:
- Console logs with file-level predictions (Normal vs WebShell)
- Summary statistics of scanned, skipped, and detected files
"""

import base64
import re
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import os


# Configuration class (must match training configuration)
class Config:
    def __init__(self):
        self.max_len = 256
        self.batch_size = 32
        self.hidden_size = 128
        self.bert_path = "../codebert/xss+webshellModel.pt"
        self.device = torch.device('cpu')


# Model class (must match training architecture exactly)
class BERTLSTM(nn.Module):
    def __init__(self, config):
        super(BERTLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(
            config.bert_path,
            local_files_only=True
        )
        for param in self.bert.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=config.hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(config.hidden_size * 2, 2)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)

        if self.lstm.bidirectional:
            forward_output = lstm_output[:, -1, :self.lstm.hidden_size]
            backward_output = lstm_output[:, 0, self.lstm.hidden_size:]
            combined_output = torch.cat((forward_output, backward_output), dim=1)
        else:
            combined_output = lstm_output[:, -1, :]

        output = self.dropout(combined_output)
        logits = self.classifier(output)
        return logits


def is_likely_base64(content):
    """Check if the content is likely Base64-encoded"""
    base64_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r")
    content_chars = set(content)

    if not content_chars.issubset(base64_chars):
        return False

    cleaned = re.sub(r'\s+', '', content)
    if len(cleaned) % 4 != 0:
        return False

    try:
        decoded = base64.b64decode(cleaned)
        if len(decoded) > 0:
            return True
    except:
        pass

    return False


def clean_base64(content):
    """Clean Base64 string by removing whitespace and fixing padding"""
    cleaned = re.sub(r'\s+', '', content)
    if len(cleaned) % 4 != 0:
        cleaned += '=' * (4 - len(cleaned) % 4)
    return cleaned


def predict_file_content(file_path, model, tokenizer, config):
    """Predict the content of a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()

        if not content:
            return {"error": "File is empty", "skipped": True}

        if not is_likely_base64(content):
            return {"message": "File is not Base64-encoded", "skipped": True}

        cleaned_base64 = clean_base64(content)

        try:
            decoded = base64.b64decode(cleaned_base64).decode('utf-8', errors='ignore')
        except:
            return {"error": "Base64 decoding failed", "skipped": True}

        text = re.sub(r'[^\w\s]', ' ', decoded).upper()
        if not text:
            return {"error": "No valid text after decoding", "skipped": True}

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=config.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = model(
                encoding['input_ids'].to(config.device),
                encoding['attention_mask'].to(config.device)
            )
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs).item()

        return {
            'prediction': pred,
            'confidence': probs[0][pred].item(),
            'opcode_preview': text[:200] + '...' if len(text) > 200 else text,
            'is_webshell': bool(pred == 1),
            'skipped': False
        }

    except Exception as e:
        return {"error": f"Error during prediction: {str(e)}", "skipped": True}


def main():
    # Initialize config and model
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.bert_path, local_files_only=True)
    model = BERTLSTM(config).to(config.device)
    model.load_state_dict(torch.load('best_model.pth', map_location=config.device))
    model.eval()

    # User input: folder path
    folder_path = input("Enter the folder path to scan: ")

    total_files = 0
    skipped_files = 0
    webshell_files = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.php'):
                file_path = os.path.join(root, file)
                total_files += 1

                print(f"\nScanning file: {file_path}")
                result = predict_file_content(file_path, model, tokenizer, config)

                if result.get('skipped', False):
                    if 'message' in result:
                        print(f"Skipped: {result['message']}")
                    elif 'error' in result:
                        print(f"Error: {result['error']}")
                    skipped_files += 1
                else:
                    print(f"Result: {'WebShell (malicious)' if result['is_webshell'] else 'Normal file'}")
                    print(f"Confidence: {result['confidence']:.2%}")
                    print(f"Opcode preview: {result['opcode_preview']}")
                    if result['is_webshell']:
                        webshell_files += 1

    # Print summary
    print("\n=== Scan Summary ===")
    print(f"Total PHP files: {total_files}")
    print(f"Skipped files: {skipped_files} (not Base64-encoded)")
    print(f"Scanned files: {total_files - skipped_files}")
    print(f"Detected WebShells: {webshell_files}")
    print("Scan complete!")


if __name__ == '__main__':
    main()
