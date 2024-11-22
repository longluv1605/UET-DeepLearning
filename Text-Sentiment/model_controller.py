import torch
import tqdm


# Train model
def train(
    model, criterion, optimizer, train_loader, device=torch.device("cpu"), epochs=1
):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm.tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Clear gradient
                optimizer.zero_grad()

                # Forward
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            print(
                f"|----> Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
            )

    return running_loss / len(train_loader)


# Evaluate model
def evaluate(model, test_loader, device=torch.device("cpu")):
    model.to(device)
    # Evaluate model on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


# Save model
def save(model, filepath):
    torch.save(model.state_dict(), filepath)
