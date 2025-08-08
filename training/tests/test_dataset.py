import os
import json
import tempfile
import torch
from PIL import Image
from training.dataset import IDDataset, collate_fn

class DummyProcessor:
    def __init__(self):
        self.tokenizer = self.DummyTokenizer()
    def __call__(self, image, return_tensors=None):
        # Return dummy pixel tensor with shape (1, 3, 64, 64)
        return type('obj', (object,), {'pixel_values': torch.rand(1, 3, 64, 64)})()
    class DummyTokenizer:
        pad_token_id = 0
        def __call__(self, text, max_length, padding, truncation, return_tensors):
            # Return dummy input_ids tensor (padded to max_length)
            input_ids = torch.randint(1, 10, (1, max_length))
            return type('obj', (object,), {'input_ids': input_ids})()
        def batch_decode(self, token_ids, skip_special_tokens=True):
            # Return dummy strings
            return ["decoded string"] * len(token_ids)

def test_iddataset_basic_functionality():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple dummy images and JSON labels
        for i in range(3):
            img_path = os.path.join(tmpdir, f"sample_{i}.png")
            Image.new("RGB", (64, 64)).save(img_path)
            label_path = os.path.join(tmpdir, f"sample_{i}.json")
            with open(label_path, "w") as f:
                json.dump({"name": f"Name{i}", "dob": f"199{i}-01-01"}, f)

        processor = DummyProcessor()
        dataset = IDDataset(tmpdir, processor, max_length=10)

        # Length should be 3
        assert len(dataset) == 3

        for i in range(len(dataset)):
            sample = dataset[i]
            assert "pixel_values" in sample and "labels" in sample
            assert sample["pixel_values"].shape[0] == 3  # RGB channels
            assert sample["labels"].shape[0] == 10  # max_length
            assert sample["labels"].dtype == torch.long

def test_iddataset_missing_label_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Image but no JSON label
        img_path = os.path.join(tmpdir, "sample_0.png")
        Image.new("RGB", (64, 64)).save(img_path)

        processor = DummyProcessor()
        dataset = IDDataset(tmpdir, processor, max_length=10)

        # Depending on dataset behavior:
        # It may include image but fail on access, or exclude image.
        # Adjust test accordingly.
        # Here, we test it includes the image but raises when accessing missing label:
        assert len(dataset) == 1

        import pytest
        with pytest.raises((FileNotFoundError, KeyError, ValueError, json.JSONDecodeError)):
            _ = dataset[0]

def test_iddataset_corrupt_json(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "sample_0.png")
        Image.new("RGB", (64, 64)).save(img_path)
        label_path = os.path.join(tmpdir, "sample_0.json")
        with open(label_path, "w") as f:
            f.write("not a valid json")

        processor = DummyProcessor()
        # Accessing sample should raise error or handle gracefully
        dataset = IDDataset(tmpdir, processor, max_length=10)
        import pytest
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _ = dataset[0]

def test_iddataset_empty_label(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "sample_0.png")
        Image.new("RGB", (64, 64)).save(img_path)
        label_path = os.path.join(tmpdir, "sample_0.json")
        with open(label_path, "w") as f:
            json.dump({}, f)  # Empty dict as label

        processor = DummyProcessor()
        dataset = IDDataset(tmpdir, processor, max_length=10)

        sample = dataset[0]
        assert "pixel_values" in sample and "labels" in sample
        assert sample["labels"].shape[0] == 10

def test_collate_fn_varied_lengths():
    processor = DummyProcessor()

    # Create samples with different label lengths (simulate padding -100)
    sample1 = {
        "pixel_values": torch.rand(3, 64, 64),
        "labels": torch.tensor([1, 2, 3, -100, -100])
    }
    sample2 = {
        "pixel_values": torch.rand(3, 64, 64),
        "labels": torch.tensor([1, 2, -100, -100, -100])
    }

    batch = [sample1, sample2]
    collated = collate_fn(batch)

    # Batch dim
    assert collated["pixel_values"].shape[0] == 2
    assert collated["labels"].shape[0] == 2

    # Labels should be padded correctly (still 5 here)
    assert collated["labels"].shape[1] == 5

def test_collate_fn_empty_batch():
    collated = collate_fn([])
    # Should return empty tensors or handle gracefully
    assert "pixel_values" in collated
    assert "labels" in collated
    # Also check shape for sanity (batch dim=0)
    assert collated["pixel_values"].shape[0] == 0
    assert collated["labels"].shape[0] == 0
