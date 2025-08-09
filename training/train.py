import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from config import TrainingConfig
from dataset import IDDataset, collate_fn
from metrics import (
    calculate_token_accuracy,
    calculate_sequence_accuracy,
    calculate_average_edit_distance
)


def freeze_and_unfreeze_layers(model, unfreeze_encoder_layers=2, unfreeze_decoder_layers=2):
    """Freeze most layers, unfreeze last N in encoder/decoder."""
    # Freeze encoder
    for _, module in model.encoder.named_modules():
        for param in module.parameters():
            param.requires_grad = False

    # Unfreeze last N encoder layers
    encoder_layers = list(model.encoder.named_modules())
    for _, module in encoder_layers[-unfreeze_encoder_layers:]:
        for param in module.parameters():
            param.requires_grad = True

    # Freeze decoder
    for layer in model.decoder.model.decoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # Unfreeze last N decoder layers
    for layer in model.decoder.model.decoder.layers[-unfreeze_decoder_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Keep embeddings trainable
    for param in model.decoder.model.decoder.embed_positions.parameters():
        param.requires_grad = True
    for param in model.decoder.model.decoder.embed_tokens.parameters():
        param.requires_grad = True


def train():
    wandb.init(project="ID_document_parser", config={k: v for k, v in TrainingConfig.__dict__.items() if not k.startswith("__")})


    processor = DonutProcessor.from_pretrained(TrainingConfig.MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(TrainingConfig.MODEL_NAME)

    # Set token IDs
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = TrainingConfig.MAX_LENGTH

    freeze_and_unfreeze_layers(model, unfreeze_encoder_layers=2, unfreeze_decoder_layers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = IDDataset(TrainingConfig.TRAIN_DIR, processor)
    val_dataset = IDDataset(TrainingConfig.VAL_DIR, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=TrainingConfig.LEARNING_RATE
    )

    for epoch in range(TrainingConfig.MAX_EPOCHS):
        if epoch == 3:  # unfreeze more later
            freeze_and_unfreeze_layers(model, unfreeze_encoder_layers=3, unfreeze_decoder_layers=3)

        # --- Training ---
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
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

        # --- Validation ---
        model.eval()
        val_loss, total_token_acc, total_seq_acc, total_edit_dist = 0, 0, 0, 0
        total_samples, low_confidence_count = 0, 0

        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

                generated_ids = model.generate(
                    pixel_values,
                    max_length=TrainingConfig.MAX_LENGTH,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    early_stopping=True
                )

                pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id
                true_texts = processor.batch_decode(labels_for_decode, skip_special_tokens=True)

                token_acc = calculate_token_accuracy(generated_ids, labels_for_decode, processor.tokenizer.pad_token_id)
                total_token_acc += token_acc * pixel_values.size(0)

                seq_acc = calculate_sequence_accuracy(pred_texts, true_texts)
                total_seq_acc += seq_acc * pixel_values.size(0)

                avg_edit_dist = calculate_average_edit_distance(pred_texts, true_texts)
                total_edit_dist += avg_edit_dist * pixel_values.size(0)

                probs = F.softmax(outputs.logits, dim=-1)
                max_probs, _ = torch.max(probs, dim=-1)
                low_conf_flags = (max_probs < TrainingConfig.CONFIDENCE_THRESHOLD).any(dim=1)
                low_confidence_count += low_conf_flags.sum().item()

                total_samples += pixel_values.size(0)

        avg_val_loss = val_loss / len(val_loader)
        avg_token_acc = total_token_acc / total_samples
        avg_seq_acc = total_seq_acc / total_samples
        avg_edit_dist = total_edit_dist / total_samples
        low_conf_rate = low_confidence_count / total_samples

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Token Accuracy: {avg_token_acc:.4f}")
        print(f"Sequence Accuracy: {avg_seq_acc:.4f}")
        print(f"Avg Edit Distance: {avg_edit_dist:.4f}")
        print(f"Low Confidence Rate (<{TrainingConfig.CONFIDENCE_THRESHOLD}): {low_conf_rate:.4f}")

        wandb.log({
            "val_loss": avg_val_loss,
            "val_token_accuracy": avg_token_acc,
            "val_sequence_accuracy": avg_seq_acc,
            "val_avg_edit_distance": avg_edit_dist,
            "val_low_confidence_rate": low_conf_rate,
        })

        # Save checkpoint
        checkpoint_path = os.path.join(TrainingConfig.CHECKPOINT_DIR, f"epoch_{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)


if __name__ == "__main__":
    train()
