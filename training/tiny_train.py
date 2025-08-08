import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import DonutProcessor, VisionEncoderDecoderModel
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from .config import TrainingConfig
from .dataset import IDDataset, collate_fn
from .metrics import calculate_token_accuracy, calculate_sequence_accuracy, calculate_average_edit_distance

def freeze_and_unfreeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False

    encoder_layers = []
    for name, module in model.encoder.named_modules():
        if module.__class__.__name__ == "DonutSwinLayer":
            encoder_layers.append(module)
    for layer in encoder_layers[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    decoder_layers = list(model.decoder.model.decoder.layers)
    for layer in decoder_layers[-2:]:
        for param in layer.parameters():
            param.requires_grad = True

    for param in model.decoder.model.decoder.embed_positions.parameters():
        param.requires_grad = True
    for param in model.decoder.model.decoder.embed_tokens.parameters():
        param.requires_grad = True

def train_tiny_subset():
    wandb.init(project="ID_document_parser_tiny_trial")

    processor = DonutProcessor.from_pretrained(TrainingConfig.MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(TrainingConfig.MODEL_NAME)
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    freeze_and_unfreeze_layers(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    full_train_dataset = IDDataset(TrainingConfig.TRAIN_DIR, processor)
    full_val_dataset = IDDataset(TrainingConfig.VAL_DIR, processor)

    # Use only first 5 samples for quick trial run
    tiny_train_dataset = Subset(full_train_dataset, list(range(min(5, len(full_train_dataset)))))
    tiny_val_dataset = Subset(full_val_dataset, list(range(min(5, len(full_val_dataset)))))

    train_loader = DataLoader(tiny_train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(tiny_val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=TrainingConfig.LEARNING_RATE)

    # Just 1 epoch for trial
    for epoch in range(1):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc="Training (tiny subset)"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        wandb.log({"train_loss": avg_train_loss, "epoch": epoch + 1})
        print(f"Epoch {epoch+1} - Avg Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss, total_token_acc, total_seq_acc, total_edit_dist = 0, 0, 0, 0
        total_samples, low_confidence_count = 0, 0

        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                max_probs, pred_tokens = torch.max(probs, dim=-1)

                token_acc = calculate_token_accuracy(pred_tokens, labels, processor.tokenizer.pad_token_id)
                total_token_acc += token_acc * pixel_values.size(0)

                low_confidence_flags = (max_probs < TrainingConfig.CONFIDENCE_THRESHOLD).any(dim=1)
                low_confidence_count += low_confidence_flags.sum().item()

                pred_texts = processor.batch_decode(pred_tokens, skip_special_tokens=True)
                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id
                true_texts = processor.batch_decode(labels_for_decode, skip_special_tokens=True)

                seq_acc = calculate_sequence_accuracy(pred_texts, true_texts)
                total_seq_acc += seq_acc * pixel_values.size(0)

                avg_edit_dist = calculate_average_edit_distance(pred_texts, true_texts)
                total_edit_dist += avg_edit_dist * pixel_values.size(0)

                total_samples += pixel_values.size(0)

        avg_val_loss = val_loss / len(val_loader)
        avg_token_acc = total_token_acc / total_samples
        avg_seq_acc = total_seq_acc / total_samples
        avg_edit_dist = total_edit_dist / total_samples
        low_confidence_rate = low_confidence_count / total_samples

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Token Accuracy: {avg_token_acc:.4f}")
        print(f"Sequence Accuracy: {avg_seq_acc:.4f}")
        print(f"Avg Edit Distance: {avg_edit_dist:.4f}")
        print(f"Low Confidence Rate (<{TrainingConfig.CONFIDENCE_THRESHOLD}): {low_confidence_rate:.4f}")

        wandb.log({
            "val_loss": avg_val_loss,
            "val_token_accuracy": avg_token_acc,
            "val_sequence_accuracy": avg_seq_acc,
            "val_avg_edit_distance": avg_edit_dist,
            "val_low_confidence_rate": low_confidence_rate,
        })

if __name__ == "__main__":
    train_tiny_subset()
