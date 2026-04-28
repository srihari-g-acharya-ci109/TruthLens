"""
TruthLens - Model Training Script
Trains the LSTM model on the Kaggle Fake News Dataset.

Usage:
    python train.py

Prerequisites:
    1. Run 'python download_dataset.py' first to download the dataset
    2. Install dependencies: pip install -r requirements.txt
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model.lstm_model import LSTMClassifier
from model.preprocess import TextPreprocessor

# === Configuration ===
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data')
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 100
VOCAB_SIZE = 25000
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
N_LAYERS = 2
DROPOUT = 0.3
TEST_SIZE = 0.2
RANDOM_STATE = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset():
    """Load the Custom Fake News Dataset."""
    print("[1/6] Loading dataset...")

    dataset_path = os.path.join(DATASET_PATH, 'dataset.csv')

    if not os.path.exists(dataset_path):
        print(f"  Dataset not found at {dataset_path}")
        print("  Please place your dataset file named 'dataset.csv' inside the backend/data directory.")
        return None, None

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Ensure required columns exist
    required_cols = ['title', 'text', 'label']
    for col in required_cols:
        if col not in df.columns:
            print(f"  Error: Missing required column '{col}' in the dataset.")
            return None, None

    # Combine title and text
    df['text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
    
    # Shuffle and drop empty texts
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df = df.dropna(subset=['text'])

    print(f"  Total samples: {len(df)}")
    print(f"  Fake (1): {len(df[df.label == 1])}, Real (0): {len(df[df.label == 0])}")

    return df['text'].values, df['label'].values


def build_vocab(texts, preprocessor, max_vocab=VOCAB_SIZE):
    """Build vocabulary from preprocessed texts."""
    print("[2/6] Building vocabulary...")

    counter = Counter()
    for i, text in enumerate(texts):
        tokens = preprocessor.preprocess_to_tokens(text)
        counter.update(tokens)
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(texts)} texts...")

    # Reserve 0=PAD, 1=UNK
    most_common = counter.most_common(max_vocab - 2)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, _) in enumerate(most_common, 2):
        vocab[word] = idx

    print(f"  Vocabulary size: {len(vocab)}")
    return vocab


def texts_to_tensors(texts, vocab, preprocessor, max_len=MAX_SEQ_LEN):
    """Convert list of texts to padded index tensors."""
    print("[3/6] Converting texts to tensors...")

    all_indices = []
    for i, text in enumerate(texts):
        tokens = preprocessor.preprocess_to_tokens(text)
        indices = [vocab.get(t, 1) for t in tokens]  # 1 = UNK

        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices += [0] * (max_len - len(indices))

        all_indices.append(indices)

        if (i + 1) % 5000 == 0:
            print(f"  Converted {i + 1}/{len(texts)} texts...")

    return torch.tensor(all_indices, dtype=torch.long)


def create_data_loader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    """Create a DataLoader from tensors."""
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i, (batch_X, batch_y) in enumerate(loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()

        output = model(batch_X).squeeze(1)
        loss = criterion(output, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        predicted = (output > 0.5).float()
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
        
        if (i + 1) % 100 == 0:
            print(f"    Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total * 100


def evaluate(model, loader, criterion):
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X).squeeze(1)
            loss = criterion(output, batch_y)

            total_loss += loss.item()
            predicted = (output > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    return avg_loss, accuracy, all_preds, all_labels


def main():
    print("=" * 60)
    print("  TruthLens — LSTM Model Training")
    print("=" * 60)
    print(f"  Device: {device}")
    start_time = time.time()

    # Load data
    texts, labels = load_dataset()
    if texts is None:
        return

    # Preprocess
    preprocessor = TextPreprocessor()

    # Build vocab
    vocab = build_vocab(texts, preprocessor)

    # Split data
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"  Train: {len(X_train_text)}, Test: {len(X_test_text)}")

    # Convert to tensors
    X_train = texts_to_tensors(X_train_text, vocab, preprocessor)
    X_test = texts_to_tensors(X_test_text, vocab, preprocessor)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create data loaders
    train_loader = create_data_loader(X_train, y_train_tensor)
    test_loader = create_data_loader(X_test, y_test_tensor, shuffle=False)

    # Initialize model
    print("[4/6] Initializing LSTM model...")
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=1,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Training loop
    print("[5/6] Training...")
    history = {
        'epochs': [], 'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': []
    }
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step(val_loss)

        history['epochs'].append(epoch)
        history['train_loss'].append(round(train_loss, 4))
        history['val_loss'].append(round(val_loss, 4))
        history['train_accuracy'].append(round(train_acc, 2))
        history['val_accuracy'].append(round(val_acc, 2))

        print(f"  Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'truthlens_lstm.pth'))

    # Final evaluation
    print("[6/6] Final evaluation...")
    _, _, all_preds, all_labels = evaluate(model, test_loader, criterion)

    acc = accuracy_score(all_labels, all_preds) * 100
    prec = precision_score(all_labels, all_preds) * 100
    rec = recall_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds) * 100
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n  Final Results:")
    print(f"    Accuracy:  {acc:.2f}%")
    print(f"    Precision: {prec:.2f}%")
    print(f"    Recall:    {rec:.2f}%")
    print(f"    F1 Score:  {f1:.2f}%")
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"      FN={cm[1][0]}  TP={cm[1][1]}")

    # Save everything
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save vocab
    with open(os.path.join(SAVE_DIR, 'vocab.json'), 'w') as f:
        json.dump(vocab, f)

    # Save metrics
    metrics = {
        "accuracy": round(acc, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1_score": round(f1, 2),
        "confusion_matrix": {
            "true_positive": int(cm[1][1]),
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0])
        },
        "training_samples": len(X_train_text),
        "test_samples": len(X_test_text),
        "model_architecture": {
            "type": "Bidirectional LSTM",
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": N_LAYERS,
            "dropout": DROPOUT,
            "max_sequence_length": MAX_SEQ_LEN
        },
        "dataset": "Kaggle Fake News Dataset",
        "mode": "trained"
    }

    with open(os.path.join(SAVE_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save training history
    with open(os.path.join(SAVE_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed / 60:.1f} minutes!")
    print(f"  Model saved to {SAVE_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
