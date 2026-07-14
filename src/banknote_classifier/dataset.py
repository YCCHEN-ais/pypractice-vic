import torch
from torch.utils.data import Dataset

class BanknoteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["10_taka", "20_taka"]
        self.class_to_idx = {"10_taka": 0, "20_taka": 1}
        self.num_samples = 32

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mock_image = torch.randn(3, 224, 224)
        mock_label = idx % 2
        return mock_image, mock_label
