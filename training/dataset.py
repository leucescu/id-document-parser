import os
import json
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class IDDataset(Dataset):
    def __init__(self, img_dir, processor, max_length=128):
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
        self.label_paths = [p.replace(".png", ".json") for p in self.img_paths]
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        with open(self.label_paths[idx], "r") as f:
            label_data = json.load(f)
        label_str = "; ".join(f"{k}: {v}" for k, v in label_data.items())
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            label_str,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

def collate_fn(batch):
    if len(batch) == 0:
        return {
            "pixel_values": torch.empty(0, 3, 64, 64),
            "labels": torch.empty(0, 0, dtype=torch.long),
        }
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}