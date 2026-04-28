"""
TruthLens - Text Preprocessing Module
NLP pipeline for cleaning and tokenizing news article text.
Uses NLTK for tokenization, stop-word removal, and text normalization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """
    NLP text preprocessing pipeline for fake news detection.

    Steps:
        1. Lowercase conversion
        2. URL removal
        3. HTML tag removal
        4. Special character and number removal
        5. Tokenization (NLTK word_tokenize)
        6. Stop-word removal
        7. Lemmatization
        8. Rejoin tokens
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        # Keep some negation words that are important for sentiment/fake detection
        self.important_stopwords = {'not', 'no', 'nor', 'neither', 'never', 'none',
                                     'nobody', 'nothing', 'nowhere', 'hardly', 'barely'}
        self.stop_words -= self.important_stopwords

    def clean_text(self, text: str) -> str:
        """Apply full preprocessing pipeline to raw text."""
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters and digits (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> list:
        """Tokenize cleaned text using NLTK word_tokenize."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list) -> list:
        """Remove stop words from token list."""
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def lemmatize(self, tokens: list) -> list:
        """Apply lemmatization to tokens."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess(self, text: str) -> str:
        """Full pipeline: clean → tokenize → remove stopwords → lemmatize → rejoin."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return ' '.join(tokens)

    def preprocess_to_tokens(self, text: str) -> list:
        """Full pipeline returning token list instead of string."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens

    def get_preprocessing_steps(self, text: str) -> dict:
        """Return intermediate results of each preprocessing step for visualization."""
        original = text
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        after_stopwords = self.remove_stopwords(tokens)
        after_lemma = self.lemmatize(after_stopwords)
        final = ' '.join(after_lemma)

        return {
            "original": original[:500],
            "after_cleaning": cleaned[:500],
            "tokens": tokens[:50],
            "after_stopword_removal": after_stopwords[:50],
            "after_lemmatization": after_lemma[:50],
            "final_text": final[:500],
            "original_word_count": len(text.split()),
            "final_word_count": len(after_lemma),
            "words_removed": len(text.split()) - len(after_lemma)
        }
