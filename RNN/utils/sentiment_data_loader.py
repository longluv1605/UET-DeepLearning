import torch
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string

from sklearn.model_selection import train_test_split
from collections import Counter        

LANGUAGE='english'


# Load requirements
def load_requirements():
    # Init neccessary tools
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("wordnet")

# Sentence preprocessing function
def preprocess_text(text):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words(LANGUAGE))
    
    text = text.lower()  # Convert into lower case
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens if word not in string.punctuation
    ]  # Remove punctuation
    tokens = [
        word for word in tokens if word not in stop_words
    ]  # Remove stopwords
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lematizing
    return " ".join(tokens)

# Build vocab
def build_vocab(texts, max_vocab_size=10000):
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    vocab = {
        word: idx + 2
        for idx, (word, _) in enumerate(word_counts.most_common(max_vocab_size))
    }
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

# Indexing
def encode_text(text, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()]

# Encode padding for train and test set
def pad_sequences(sequences, max_length):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_length:
            padded_sequences.append(seq[:max_length])
        elif len(seq) < max_length:
            padded_sequences.append(seq + [0] * (max_length - len(seq)))
        else:
            padded_sequences.append(seq)
    return torch.Tensor(padded_sequences)

def load_dataset(dataframe, max_length):
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe["review"], dataframe["sentiment"], test_size=0.2, random_state=42
    )

    # Build vocab
    vocab = build_vocab(X_train)

    # Encode and padding
    X_train_encoded = [encode_text(text, vocab) for text in X_train]
    X_test_encoded = [encode_text(text, vocab) for text in X_test]
    X_train_padded = pad_sequences(X_train_encoded, max_length)
    X_test_padded = pad_sequences(X_test_encoded, max_length)

    # Convert into Tensor
    y_train = torch.tensor([1 if label == "positive" else 0 for label in y_train])
    y_test = torch.tensor([1 if label == "positive" else 0 for label in y_test])
    
    train_dataset = TextDataset(X_train_padded, y_train)
    test_dataset = TextDataset(X_test_padded, y_test)

    return vocab, train_dataset, test_dataset


# Define dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.long()
        self.labels = labels.long()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]