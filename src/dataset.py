from torch.utils.data import Dataset
import torch


class AirlineDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.encodings['input_ids'])
        target_ids = torch.tensor(self.labels[idx])
        return {"input_ids": input_ids, "labels": target_ids}

    def __len__(self):
        return len(self.labels)

