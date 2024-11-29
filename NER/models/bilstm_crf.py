import torch.nn as nn
from TorchCRF import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden_dim = hidden_dim
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, words, tags=None, mask=None):
        embeddings = self.embedding(words)
        lstm_out, _ = self.lstm(embeddings)
        logits = self.hidden2tag(lstm_out)
        
        if mask is not None:
            mask = mask.bool()
        
        if tags is not None:
            loss = -self.crf(logits, tags, mask=mask, reduction="mean")
            return loss
        else:
            predictions = self.crf.decode(logits, mask=mask)
            return predictions