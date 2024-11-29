import torch
from torch.utils.data import Dataset, DataLoader


def read_data(file_path):
    sentences, sentence = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                word, pos, chunk, ner = line.split()
                sentence.append((word, pos, chunk, ner))
        if sentence:
            sentences.append(sentence)  # Append last sentence
    return sentences


class NERDataset(Dataset):
    def __init__(self, sentences, word2idx, tag2idx, max_len):
        self.sentences = sentences
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        words = [word for word, _, _, _ in sentence]
        tags = [tag for _, _, _, tag in sentence]

        word_indices = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
        tag_indices = [self.tag2idx.get(t, self.tag2idx["O"]) for t in tags]

        # Padding
        word_indices = word_indices[:self.max_len] + [self.word2idx["<PAD>"]] * (self.max_len - len(word_indices))
        tag_indices = tag_indices[:self.max_len] + [self.tag2idx["O"]] * (self.max_len - len(tag_indices))

        mask = [1 if i != self.word2idx["<PAD>"] else 0 for i in word_indices]

        return torch.tensor(word_indices), torch.tensor(tag_indices), torch.tensor(mask)
    
    
def create_vocab(sentences):
    words = set()
    tags = set()
    for sentence in sentences:
        for word, _, _, tag in sentence:
            words.add(word)
            tags.add(tag)
    word2idx = {word: idx for idx, word in enumerate(sorted(words), start=2)}
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1
    tag2idx = {tag: idx for idx, tag in enumerate(sorted(tags))}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    return word2idx, tag2idx, idx2tag