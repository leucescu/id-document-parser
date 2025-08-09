import os
import json
import datetime
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset


def normalize_date(date_str):
    """Convert date to YYYY-MM-DD if possible."""
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return date_str.strip()  # fallback


class IDDataset(Dataset):
    def __init__(self, img_dir, processor, max_length=128):
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
        self.label_paths = [p.replace(".png", ".json") for p in self.img_paths]
        self.processor = processor
        self.max_length = max_length
        self.field_order = ["name", "dob", "id_number", "expiry"]  # fixed order

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.img_paths[idx]).convert("RGB")

        # Load structured label
        with open(self.label_paths[idx], "r") as f:
            label_data = json.load(f)

        # Build normalized label string
        parts = []
        for field in self.field_order:
            value = label_data.get(field, "").strip()
            if field in ["dob", "expiry"]:
                value = normalize_date(value)
            parts.append(f"{field}: {value}")

        label_str = "; ".join(parts)

        # Add BOS/EOS tokens for consistent start/end
        bos_token = self.processor.tokenizer.bos_token or ""
        eos_token = self.processor.tokenizer.eos_token or ""
        label_str = f"{bos_token}{label_str}{eos_token}"

        # Process image
        pixel_values = self.processor(
            image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Tokenize labels
        labels = self.processor.tokenizer(
            label_str,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # Replace padding token IDs with -100 so they’re ignored in loss
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
