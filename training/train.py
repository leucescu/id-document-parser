# src/train.py
from datasets import load_dataset, DatasetDict
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image
import json

def load_local_dataset(data_dir):
    import os
    records = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".json"):
            with open(os.path.join(data_dir, fname)) as f:
                label = json.load(f)
            img_path = os.path.join(data_dir, fname.replace(".json", ".png"))
            records.append({"image_path": img_path, "label": label})
    return records

def preprocess_function(examples):
    images = [Image.open(p).convert("RGB") for p in examples["image_path"]]
    texts = [json.dumps(label) for label in examples["label"]]
    encoding = processor(images, texts, padding="max_length", truncation=True)
    return encoding

if __name__ == "__main__":
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

    # Load synthetic data
    train = load_local_dataset("data/train")
    val = load_local_dataset("data/val")
    ds = DatasetDict({
        "train": train,
        "validation": val
    })

    ds = ds.map(preprocess_function, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./donut-id-model",
        per_device_train_batch_size=2,
        num_train_epochs=5,
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="steps"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=processor.feature_extractor,  # Donut uses feature extractor + tokenizer
    )
    trainer.train()
