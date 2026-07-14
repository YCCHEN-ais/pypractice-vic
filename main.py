import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from banknote_classifier.dataset import BanknoteDataset
from banknote_classifier.models import build_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = BanknoteDataset("./data/bangla/train")
    val_dataset = BanknoteDataset("./data/bangla/validation")

    num_classes = len(train_dataset.classes)
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    print(f"Detected classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    model_name = "resnet18"
    print(f"Building model: {model_name}")
    model = build_model(model_name, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("\n--- Starting Training ---")
    model.train()
    for images, labels in tqdm(train_loader, desc=f"Training {model_name}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        break 

    print("Training step completed successfully.")

if __name__ == "__main__":
    main()
