import torch
import matplotlib.pyplot as plt
from torchinfo import summary
import random
import logging


def inspect_images(
    class_names, dataloader: torch.utils.data.DataLoader, k: int, seed=42
):
    batch = next(iter(dataloader))
    images, labels = batch

    image_list = [img for img in images]
    label_list = [label for label in labels]
    torch.manual_seed(42)
    random.seed(42)

    combined = list(zip(image_list, label_list))
    random_samples = random.sample(combined, k)

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    for ax, (img, label) in zip(axs, random_samples):
        ax.imshow((img.permute(1, 2, 0)))
        ax.set_title(f"Label: {class_names[label]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def summarize_model(model):
    model_summary = summary(
        model=model,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        verbose=0,
        row_settings=["var_names"],
    )

    print(model_summary)


def setup_logger(log_file):
    # Lấy logger mặc định
    logger = logging.getLogger()
    
    # Kiểm tra và xóa các handler cũ
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Cấu hình lại logger
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
  