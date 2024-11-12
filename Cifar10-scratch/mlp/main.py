from mlp import MLP
from hyper_tuning import *

def main():
    model = MLP(n_classes=10, n_hidden_nodes=100, image_width=32, image_height=32, color_channels=3)
    best_params, best_lost, model = hyper_tuning(model=model, epochs=1, write_log=True, download=False)
    
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Finished >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'Best params: {best_params}')
    print(f'Best loss: {best_lost}')

if __name__ == '__main__':
    main()