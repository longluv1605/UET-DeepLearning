import torch
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string

from sklearn.model_selection import train_test_split
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

class Data_Staff():
    def __init__(self, language):
        # Init neccessary tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.MAX_SEQ_LENGTH = 256 # Max length of sentence to encode
        
    # Sentence preprocessing function
    def preprocess_text(self, text):
        text = text.lower() # Convert into lower case
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in string.punctuation] # Remove punctuation
        tokens = [word for word in tokens if word not in self.stop_words] # Remove stopwords
        tokens = [self.stemmer.stem(word) for word in tokens] # Stemming
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens] # Lematizing
        return ' '.join(tokens)

    # Build vocab
    def build_vocab(self, texts, max_vocab_size=10000):
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())
        vocab = {word: idx+2 for idx, (word, _) in enumerate(word_counts.most_common(max_vocab_size))}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        return vocab

    # Indexing
    def encode_text(self, text, vocab):
        return [vocab.get(word, vocab['<UNK>']) for word in text.split]

    # Encode padding for train and test set
    def pad_sequences(self, sequences, max_length):
        padded_sequences = []
        for seq in sequences:
            if len(seq) > max_length:
                padded_sequences.append(seq[:max_length])
            elif len(seq) < max_length:
                padded_sequences.append(seq + [0] * (max_length - len(seq)))
            else: padded_sequences.append(seq)
        return torch.Tensor(padded_sequences)

    def staff(self, dataframe):
        # Preprocess review
        dataframe['review'] = dataframe['review'].apply(self.preprocess_text)
        
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(dataframe['review'], dataframe['sentiment'], test_size=0.2, random_state=42)
        
        # Build vocab
        vocab = self.build_vocab(X_train)

        # Encode and padding
        X_train_encoded = [self.encode_text(text, vocab) for text in X_train]
        X_test_encoded = [self.encode_text(text, vocab) for text in X_test]
        X_train_padded = self.pad_sequences(X_train_encoded, self.MAX_SEQ_LENGTH)
        X_test_padded = self.pad_sequences(X_test_encoded, self.MAX_SEQ_LENGTH)

        # Convert into Tensor
        X_train_padded = torch.tensor(X_train_padded)
        X_test_padded = torch.tensor(X_test_padded)
        y_train = torch.tensor([1 if label == 'positive' else 0 for label in y_train])
        y_test = torch.tensor([1 if label == 'positive' else 0 for label in y_test])
        
        return vocab, X_train_padded, X_test_padded, y_train, y_test

# Define dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
    
def load_data(train, batch_size, num_workers):