# BERT-WebThreats-Detector  
A unified deep-learning toolkit for detecting key web threats such as WebShells and XSS.

## Overview  
This project provides a multi-task model and supporting pipelines to classify malicious web content. It focuses on:
- **WebShells** – scripts used on web servers as backdoors or remote-control tools.  
- **Cross-Site Scripting (XSS)** – malicious script injections within web pages.

Using a shared BERT (or CodeBERT) backbone, the system is able to process both raw text and Base64-encoded payloads, handle multiple tasks simultaneously, and output confidence scores, inference time, and task-specific predictions.

## Key Features  
- Accepts mixed-format inputs (plain text or Base64-encoded).  
- Unified BERT encoder with task-specific classification heads for WebShells and XSS.  
- Stratified data splits for balanced evaluation.  
- Saves best-performing model automatically and permits comparison with previous versions.  
- Console output includes precision, recall, F1, and confusion matrix.


## Installation  
```bash
git clone https://github.com/FlorenceOTATO-index/bert-webthreats-detector.git
cd bert-webthreats-detector
pip install -r requirements.txt
```

## Usage

1. **Prepare datasets**  
   Place your datasets in the project folder (e.g., `XSS_dataset.csv`, `webshell/benign.csv`, `webshell/malicious.csv`).

2. **Configure paths and hyperparameters**  
   Update dataset paths and training parameters in the configuration file.

3. **Run training**  
   `python train.py`

4. **Evaluate tasks separately**  
   `python evaluate.py --task xss`  
   `python evaluate.py --task webshell`

5. **Load the saved model for inference**  
   ```
   from detector import WebThreatsDetector

   detector = WebThreatsDetector(model_path="best_model.pt")
   result = detector.predict(payload)
   print(result)
   ```

---

## Model Architecture

- **Encoder**: BERT / CodeBERT pre-trained model  
- **Shared Layer**: Linear → LayerNorm → ReLU → Dropout  
- **Heads**:  
  - `xss_head` → benign vs malicious XSS  
  - `webshell_head` → benign vs malicious WebShell  
- **Task Router**: Learns to decide which head to apply based on embeddings  

---

## Threats Covered

- **WebShell**: Patterns such as `eval($_POST['cmd'])`, `base64_decode`, and known web-shell keywords (`c99`, `r57`)  
- **XSS**: Indicators like `document.write`, `fromCharCode`, and injected `<script>` payloads  

---

## Contributing & License

Contributions are welcome! Feel free to open issues or submit pull requests.  
Licensed under the MIT License — see the `LICENSE` file for details.
