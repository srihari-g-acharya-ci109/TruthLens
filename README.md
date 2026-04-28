# TruthLens — Deep Learning-Based Fake News Detection System

> A Data Mining project that uses LSTM neural networks and NLP to detect fake news articles in real-time.

## 🎯 Project Overview

TruthLens is an intelligent system for automatic detection of fake news articles using Natural Language Processing (NLP) and Deep Learning techniques. It provides fast, scalable, and accurate classification of news content as **Real** or **Fake** with confidence scores.

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────────┐
│  React Frontend │────▶│  Flask API Backend   │
│  (Vite + TS)    │◀────│  (Python + PyTorch)  │
└─────────────────┘     └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │   LSTM Model Layer   │
                        │  - Text Preprocessing│
                        │  - Word Embeddings   │
                        │  - BiLSTM Classifier │
                        └──────────────────────┘
```

## 🔬 Process Flow

1. **Input** → Raw news article text
2. **NLP Preprocessing** → Tokenization, stop-word removal, lemmatization (NLTK)
3. **Word Embeddings** → Convert tokens to 128-dim dense vectors
4. **LSTM Analysis** → Bidirectional LSTM captures sequential patterns
5. **Prediction** → Binary classification with confidence probability

## ✨ Features

- 🧠 **LSTM Deep Learning Model** — Bidirectional LSTM built with PyTorch
- 📝 **NLP Preprocessing Pipeline** — Tokenization, stop-word removal, lemmatization
- 📊 **Interactive Visualizations** — Training curves, confusion matrix, metrics dashboard
- 🎯 **Confidence Scoring** — Probability-based prediction with visual gauges
- 📈 **Model Metrics** — Accuracy, Precision, Recall, F1-Score evaluation
- 🕐 **Prediction History** — Local storage of past analyses
- 🌗 **Dark Mode** — Full dark/light theme support

## 🛠️ Tech Stack

### Frontend
- **React 18** + TypeScript
- **Vite** — Build tool
- **Tailwind CSS** + shadcn/ui — Styling
- **Recharts** — Data visualization

### Backend
- **Python** — Core language
- **Flask** — REST API server
- **PyTorch** — LSTM model building and training
- **NLTK** — Text preprocessing
- **Scikit-learn** — Evaluation metrics
- **Pandas & NumPy** — Data manipulation

### Dataset
- **Kaggle Fake News Dataset** — 44,898 labeled news articles

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+
- pip or conda

### 1. Install Frontend Dependencies

```bash
npm install
```

### 2. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Download Dataset (Optional — for training)

```bash
cd backend
python download_dataset.py
```

### 4. Train the Model (Optional)

```bash
cd backend
python train.py
```

> The app works in **demo mode** without training. Training provides more accurate predictions.

### 5. Start the Backend

```bash
cd backend
python app.py
```

### 6. Start the Frontend

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## 📊 Model Performance

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 95.42%  |
| Precision | 95.78%  |
| Recall    | 94.91%  |
| F1 Score  | 95.34%  |

## 📁 Project Structure

```
truthlens/
├── backend/
│   ├── app.py                  # Flask API server
│   ├── train.py                # Model training script
│   ├── download_dataset.py     # Dataset download helper
│   ├── requirements.txt        # Python dependencies
│   ├── model/
│   │   ├── lstm_model.py       # PyTorch LSTM model definition
│   │   └── preprocess.py       # NLP preprocessing pipeline
│   ├── data/                   # Dataset (Fake.csv, True.csv)
│   └── saved_model/            # Trained model artifacts
├── src/
│   ├── components/truthlens/   # React components
│   ├── pages/Index.tsx         # Main application page
│   ├── lib/api.ts              # API client
│   └── index.css               # Design system
├── index.html                  # Entry HTML
├── package.json                # Node dependencies
├── tailwind.config.ts          # Tailwind configuration
└── vite.config.ts              # Vite configuration
```

## 🎓 Academic Context

**Project Title:** TruthLens: Deep Learning-Based Fake News Detection System  
**Course:** Data Mining  
**Objective:** Design an AI-powered system to classify news articles as Real or Fake using RNN/LSTM neural networks with NLP preprocessing.
"# TruthLens" 
