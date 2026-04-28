"""
TruthLens - LSTM Model Definition
Deep Learning model for fake news classification using PyTorch.
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM-based binary classifier for fake news detection.

    Architecture:
        - Embedding Layer: Converts word indices to dense vectors
        - LSTM Layer: Captures sequential/contextual patterns in text
        - Dropout: Regularization to prevent overfitting
        - Fully Connected Layer: Maps LSTM output to binary prediction
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1,
                 n_layers=2, dropout=0.3, pad_idx=0):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, text_lengths=None):
        # text: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))

        if text_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed)
        else:
            _, (hidden, cell) = self.lstm(embedded)

        # Concatenate final forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return self.sigmoid(output)
