import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from transformers import DonutProcessor, VisionEncoderDecoderModel
import random

# Load model and processor (adjust model_name/path if needed)
model_name = "checkpoints/epoch_5"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def enhance_image_brightness(image, factor=1.5):
    """Simple augmentation: adjust brightness."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def generate_prediction(model, processor, image, device, decoding_strategy="greedy", num_beams=3, max_length=128):
    """Run model.generate with different decoding strategies."""
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    if decoding_strategy == "greedy":
        generated_ids = model.generate(
            pixel_values,
            max_length=max_length,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            early_stopping=True
        )
    elif decoding_strategy == "beam":
        generated_ids = model.generate(
            pixel_values,
            num_beams=num_beams,
            max_length=max_length,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            early_stopping=True
        )
    else:
        raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")

    pred_str = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_ids, pred_str, pixel_values

def get_token_confidence(model, pixel_values, generated_ids):
    """
    Calculate max token probabilities for generated tokens.
    generated_ids shape: (1, seq_len), includes decoder_start_token_id at position 0
    For teacher forcing, shift right the generated_ids by 1:
    decoder_input_ids = generated_ids[:, :-1]
    Labels = generated_ids[:, 1:]
    """
    with torch.no_grad():
        decoder_input_ids = generated_ids[:, :-1]
        outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
    logits = outputs.logits  # (batch_size=1, seq_len-1, vocab_size)
    probs = F.softmax(logits, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1)  # max prob per token (1, seq_len-1)
    return max_probs.squeeze(0)  # return 1D tensor

def inference_with_confidence(image_path, confidence_threshold=0.8, max_retries=2):
    """
    Runs inference on an image path with confidence checking.
    Retries with brightness augmentation if confidence is low.
    Returns prediction and whether manual review is needed.
    """
    image = Image.open(image_path).convert("RGB")

    for attempt in range(max_retries + 1):
        decoding_strategy = "greedy" if attempt == 0 else "beam"
        num_beams = 5 if decoding_strategy == "beam" else 1

        generated_ids, pred_str, pixel_values = generate_prediction(
            model, processor, image, device,
            decoding_strategy=decoding_strategy,
            num_beams=num_beams
        )

        max_probs = get_token_confidence(model, pixel_values, generated_ids)

        low_confidence = (max_probs < confidence_threshold).any().item()

        if not low_confidence:
            return pred_str, False  # confident prediction

        print(f"Low confidence detected on attempt {attempt+1}")

        if attempt < max_retries:
            factor = random.uniform(0.7, 1.3)
            image = enhance_image_brightness(image, factor=factor)

    return pred_str, True  # manual review needed


if __name__ == "__main__":
    # Example usage
    test_image = "data/validation/id_0974.png"
    prediction, manual_review = inference_with_confidence(test_image)

    print("Prediction:", prediction)
    if manual_review:
        print("Manual review recommended due to low confidence.")
    else:
        print("Prediction confident, no review needed.")
