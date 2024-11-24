from utils.data_loader import *
from models.cnn.text_cnn import TextCNN
from utils.model_controller import *
from utils.hyper_tuning import hyper_tuning

import pandas as pd
import torch
import torch.nn as nn

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    load_requirements()
    
    df = pd.read_csv('./data/imdb/review.csv')
    
    df['review'] = df['review'].apply(preprocess_text)

    MAX_SEQ_LENGTH = 256  # Max length of sentence to encode
    
    vocab, train_dataset, test_dataset = load_dataset(df, MAX_SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    
    # Init model
    vocab_size = len(vocab)
    embedding_dim = 100
    num_classes = 2  # Positive/Negative
    kernel_sizes = [3, 4, 5]
    num_filters = 100

    model = TextCNN(vocab_size, embedding_dim, num_classes, kernel_sizes, num_filters)
    
    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    train(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, device=device, epochs=5)
    
    evaluate(model=model, test_loader=test_loader, device=device)
 
    
if __name__ == "__main__":
    main()