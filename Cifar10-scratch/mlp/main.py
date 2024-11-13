from mlp import MLP
from hyper_tuning import *
from model_trainer import ModelTrainer

def main():
    model = MLP(n_classes=10, n_hidden_nodes=100, image_width=32, image_height=32, color_channels=3)
    model_trainer = ModelTrainer(model=model)
    model_trainer.train(epochs=1, write_log=True)

if __name__ == '__main__':
    main()