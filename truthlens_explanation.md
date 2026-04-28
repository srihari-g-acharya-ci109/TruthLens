# TruthLens: Deep Learning Fake News Detection System
**Project Architecture and Workflow Overview**

---

## 1. Introduction
TruthLens is an intelligent, end-to-end web application developed for our Data Mining project. Its primary goal is to classify news articles as either **Real** or **Fake** using Natural Language Processing (NLP) and Deep Learning (LSTM neural networks). 

The system provides real-time inference, offering confidence probabilities and visualizing the exact preprocessing steps the AI takes to understand the text.

---

## 2. System Architecture
TruthLens is divided into a decoupled client-server architecture:

### 🎨 Frontend (Client)
- **Tech Stack:** React 18, TypeScript, Vite, Tailwind CSS.
- **Role:** Takes user input (news text), communicates with the backend via REST API, and beautifully renders the prediction result, confidence scores, and training metrics charts using Recharts.

### 🧠 Backend (Server)
- **Tech Stack:** Python, Flask, PyTorch, NLTK.
- **Role:** Handles the heavy lifting. It exposes endpoints (`/api/predict`, `/api/metrics`) that receive text, pass it through the NLP pipeline, feed it into the PyTorch LSTM model, and return the calculated probabilities.

---

## 3. The Data Mining Pipeline

TruthLens perfectly exemplifies the KDD (Knowledge Discovery in Databases) process applied to unstructured text data.

### Step A: Data Selection & Preprocessing (NLTK)
Raw text is incredibly noisy. We utilize traditional Data Mining preprocessing techniques using the Natural Language Toolkit (NLTK):
1. **Data Cleaning:** Removal of punctuation, HTML tags, numbers, and special characters.
2. **Tokenization:** Breaking paragraphs down into granular attributes (tokens).
3. **Stop-word Removal:** Reducing dimensionality by removing frequent but uninformative words ("the", "is", "at").
4. **Lemmatization:** Converting variations of words to their base root (e.g., "running" → "run") to consolidate semantic features.

### Step B: Feature Extraction (Word Embeddings)
In Data Mining, features must be numeric. The cleaned tokens are mapped to a vocabulary dictionary (size: 25,000 words) using Frequency Analysis. The tokens are then embedded into a 128-dimensional continuous vector space. This represents words mathematically, capturing context and semantic relationships.

### Step C: Data Modeling (Bidirectional LSTM)
We use a Recurrent Neural Network architecture to mine sequential patterns from the text features:
- **LSTM (Long Short-Term Memory):** Prevents the vanishing gradient problem, allowing the model to "remember" long-term dependencies within a news article.
- **Bidirectional Mechanism:** The data is processed both forwards and backwards simultaneously, allowing the network to understand a word's meaning based on both prior and future context.

---

## 4. Dataset Mining Challenges: Bias & Out-of-Distribution Data

A core component of Data Mining is evaluating the quality and biases of your data source.

**The Training Set & Class Imbalance Behaviors:**
The model was trained on the Kaggle Fake vs. Real News Dataset (approx. 70,000 articles). During EDA (Exploratory Data Analysis), a massive correlational bias was found: the "Real" news class exclusively consisted of *Reuters* articles, while the "Fake" class consisted of highly sensationalized political blogs.

**Algorithmic Adjustment to Bias:**
Because the model overfitted to the dry linguistic patterns of Reuters, it heavily penalized local/sensational news from other sources (like Aaj Tak), incorrectly classifying them as Fake (Out-of-Distribution Data). To counteract this without retraining the entire dataset, we engineered a **Heuristic Bias Corrector** that mathematically applies a 0.65x weight penalty to the fake probability matrix, forcing it to require extreme confidence before determining an anomaly (Fake News).

---

## 5. Data Mining Evaluation Metrics

Model performance in a binary classification context is measured using standard evaluation paradigms:
- **Confusion Matrix:** Provides a foundational view of True Positives, True Negatives, False Positives (Type I error), and False Negatives (Type II error).
- **Accuracy:** General correctness of the model across the 70,000 document set.
- **Precision (Exactness):** When it flags an article as Fake, how often is it right? (Crucial to avoid censoring real journalism).
- **Recall (Completeness):** Out of all the genuinely Fake articles, how many did it successfully catch? 
- **F1-Score:** The harmonic mean of Precision and Recall.
