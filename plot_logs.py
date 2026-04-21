import json
import matplotlib.pyplot as plt
import os

def plot_logs(log_path='checkpoints/run_dino/log.txt'):
    if not os.path.exists(log_path):
        print(f"Error: Could not find {log_path}")
        return

    epochs = []
    train_loss, val_loss = [], []
    train_ce, val_ce = [], []
    train_bbox, val_bbox = [], []
    train_giou, val_giou = [], []

    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                epochs.append(data['epoch'])
                train_loss.append(data['train_loss'])
                val_loss.append(data['val_loss'])
                train_ce.append(data['train_ce'])
                val_ce.append(data['val_ce'])
                train_bbox.append(data['train_bbox'])
                val_bbox.append(data['val_bbox'])
                train_giou.append(data['train_giou'])
                val_giou.append(data['val_giou'])
            except KeyError:
                continue # Skip malformed lines

    # Create a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Total Loss
    axs[0, 0].plot(epochs, train_loss, label='Train Total', color='blue', linewidth=2)
    axs[0, 0].plot(epochs, val_loss, label='Val Total', color='orange', linewidth=2)
    axs[0, 0].set_title('Total Loss', fontsize=14)
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    axs[0, 0].legend()

    # GIoU Loss (Most critical for DETR)
    axs[0, 1].plot(epochs, train_giou, label='Train GIoU', color='green', linewidth=2)
    axs[0, 1].plot(epochs, val_giou, label='Val GIoU', color='red', linewidth=2)
    axs[0, 1].set_title('GIoU Loss (Overlap)', fontsize=14)
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    axs[0, 1].legend()

    # BBox L1 Loss
    axs[1, 0].plot(epochs, train_bbox, label='Train BBox L1', color='purple', linewidth=2)
    axs[1, 0].plot(epochs, val_bbox, label='Val BBox L1', color='brown', linewidth=2)
    axs[1, 0].set_title('BBox L1 Loss (Coordinates)', fontsize=14)
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    axs[1, 0].legend()

    # Classification Error
    axs[1, 1].plot(epochs, train_ce, label='Train Class CE', color='teal', linewidth=2)
    axs[1, 1].plot(epochs, val_ce, label='Val Class CE', color='magenta', linewidth=2)
    axs[1, 1].set_title('Classification Loss', fontsize=14)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('dino_loss_curves.png', dpi=300)
    print("Successfully generated 'dino_loss_curves.png'")

if __name__ == "__main__":
    plot_logs()