import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import pickle

def train_model(model, train_loader, val_loader, optimizer, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for words, tags, mask in tqdm(train_loader, unit='batch', desc=f"Training {epoch + 1}/{epochs}"):
            words, tags, mask = words.to(device), tags.to(device), mask.to(device)
            loss = model(words, tags, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"|----> Loss: {total_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for words, tags, mask in val_loader:
                words, tags, mask = words.to(device), tags.to(device), mask.to(device)
                val_loss += model(words, tags, mask).item()
        print(f"   |----> Validation Loss: {val_loss:.4f}")
        

def evaluate_model(model, test_loader, idx2tag, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for words, tags, mask in test_loader:
            words, tags, mask = words.to(device), tags.to(device), mask.to(device)
            predictions = model(words, mask=mask)
            for pred, true, m in zip(predictions, tags, mask):
                all_preds.extend([idx2tag[p] for p, mask_val in zip(pred, m) if mask_val])
                all_labels.extend([idx2tag[t.item()] for t, mask_val in zip(true, m) if mask_val])
    print(classification_report(all_labels, all_preds))
    
    
def evaluate_model_by_accuracy(model, test_loader, idx2tag, device):
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for words, tags, mask in test_loader:
            words, tags, mask = words.to(device), tags.to(device), mask.to(device)
            mask = mask.bool() 
            predictions = model(words, mask=mask)
            
            for i in range(len(words)):
                true_tags = tags[i][mask[i]].cpu().numpy()
                pred_tags = predictions[i]
                
                all_labels.extend(true_tags)
                all_preds.extend(pred_tags)

    all_labels_filtered = [l for l in all_labels if idx2tag[l] != 'O']
    all_preds_filtered = [p for l, p in zip(all_labels, all_preds) if idx2tag[l] != 'O']

    overall_accuracy = accuracy_score(all_labels, all_preds)
    filtered_accuracy = accuracy_score(all_labels_filtered, all_preds_filtered)

    return overall_accuracy, filtered_accuracy


def model_predict(model, inputs, idx2tag, device=torch.device('cpu'), mask=None):
    model.to(device)
    input.to(device)
    
    model.eval()
    with torch.no_grad():
        preds = model(inputs, mask=mask)
        preds = [idx2tag[pred] for pred in preds]
    
def save_model(model, model_params, idx2tag, name, mask=None):
    model_path = f'save/models/{name}.pth'
    model_params_path = f'save/models/{name}_params.pth'
        
    idx2tag_path = f'save/models/{name}_idx2tag.pkl'
    
    torch.save(model.state_dict(), model_path)
    pickle.dump(idx2tag, open(idx2tag_path, 'wb'))
    pickle.dump(model_params, open(model_params_path, 'wb'))
    
    print(f'Saved model to {model_path}')
    print(f'Saved model parameters to {model_params_path}')
    print(f'Saved model idx2tag to {idx2tag_path}')