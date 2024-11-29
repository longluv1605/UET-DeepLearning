import torch
from common.data_prepare import *
from common.model_controller import *
from models.bilstm_crf import BiLSTM_CRF

def main():
    train_file = 'data/eng/eng.train'
    val_file = 'data/eng/eng.testa'
    test_file = 'data/eng/eng.testb'

    train_sentences = read_data(train_file)
    val_sentences = read_data(val_file)
    test_sentences = read_data(test_file)


    word2idx, tag2idx, idx2tag = create_vocab(train_sentences)

    max_len = 50
    batch_size = 32

    train_dataset = NERDataset(train_sentences, word2idx, tag2idx, max_len)
    val_dataset = NERDataset(val_sentences, word2idx, tag2idx, max_len)
    test_dataset = NERDataset(test_sentences, word2idx, tag2idx, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    embedding_dim=100
    hidden_dim=128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM_CRF(len(word2idx), len(tag2idx), embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, optimizer, epochs=10, device=device)

    evaluate_model(model, test_loader, idx2tag, device)

    overall_acc, filtered_acc = evaluate_model_by_accuracy(model, test_loader, idx2tag, device)
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    print(f"Filtered Accuracy (excluding 'O' tags): {filtered_acc * 100:.2f}%")

    model_params = {
        'vocab_size': len(word2idx),
        'tagset_size': len(tag2idx),
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim
    }
    
    save_model(
        model=model,
        model_params=model_params,
        idx2tag=idx2tag,
        name=f'bilstm_{filtered_acc}'
    )
    
    
if __name__=='__main__':
    main()