import torchvision
import torch.nn as nn

def create_model(name: str, version: str = 'DEFAULT', class_names=None):
    # Get the weight class from the weight_name
    weights_class = getattr(torchvision.models, name + '_Weights', None)
    if weights_class is None:
        raise ValueError(f"Weight '{name}_Weights' is invalid or does not exist.")

    # Get the specific version of the weight (e.g., DEFAULT, IMAGENET1K_V1)
    weights = getattr(weights_class, version, None)
    if weights is None:
        raise ValueError(f"Version '{version}' is not available for '{name}_Weights'.")

    # Get the model class from the model_name string
    model_class = getattr(torchvision.models, name.lower(), None)
    if model_class is None:
        raise ValueError(f"Model '{name.lower()}' is invalid or does not exist.")

    # Initialize the model with the specified weights
    model = model_class(weights=weights)

    # Get the transform associated with the weights
    transform = weights.transforms()

    # Freeze all parameters to prevent them from being trained
    for param in model.parameters():
        param.requires_grad = False

    # Step to find the last layer (output layer)
    def find_last_layer(model):
        # Traverse through all named children in reverse order to find the last layer
        *_, last_name, last_layer = list(model.named_children())[-1]
        return last_name, last_layer

    # Find the last layer of the model
    last_name, last_layer = find_last_layer(model)

    # Replace the output layer's number of classes
    if isinstance(last_layer, nn.Sequential):
        in_features = last_layer[-1].in_features
        new_layer = nn.Linear(in_features, len(class_names)) # Replace the last layer in the model
        last_layer[-1] = new_layer
        setattr(model, last_name, last_layer)
    else:
        in_features = last_layer.in_features
        new_layer = nn.Linear(in_features, len(class_names))
        setattr(model, last_name, new_layer)  # Replace the last layer in the model

    return model, transform