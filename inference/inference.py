import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from transformers import DonutProcessor, VisionEncoderDecoderModel
import random

# Load model and processor (adjust model_name/path if needed)
model_name = "naver-clova-ix/donut-base"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def enhance_image_brightness(image, factor=1.5):
    """Simple augmentation: adjust brightness."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def generate_prediction(model, processor, image, device, decoding_strategy="greedy", num_beams=3):
    """Run model.generate with different decoding strategies."""
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    if decoding_strategy == "greedy":
        generated_ids = model.generate(pixel_values)
    elif decoding_strategy == "beam":
        generated_ids = model.generate(pixel_values, num_beams=num_beams, early_stopping=True)
    else:
        raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")

    pred_str = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_ids, pred_str, pixel_values

def get_token_confidence(model, pixel_values, generated_ids):
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, decoder_input_ids=generated_ids)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
    max_probs, _ = torch.max(probs, dim=-1)  # max prob per token (batch_size, seq_len)
    return max_probs

def inference_with_confidence(image_path, confidence_threshold=0.8, max_retries=2):
    """
    Runs inference on an image path with confidence checking.
    Retries with brightness augmentation if confidence is low.
    Returns prediction and whether manual review is needed.
    """
    image = Image.open(image_path).convert("RGB")

    for attempt in range(max_retries + 1):
        # Use generate_prediction helper with greedy decoding for initial attempts
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

        # Augment and retry, except on last attempt
        if attempt < max_retries:
            factor = random.uniform(0.7, 1.3)
            image = enhance_image_brightness(image, factor=factor)

    return pred_str, True  # manual review needed

if __name__ == "__main__":
    # Example usage
    test_image = "path/to/your/test/image.png"
    prediction, manual_review = inference_with_confidence(test_image)

    print("Prediction:", prediction)
    if manual_review:
        print("Manual review recommended due to low confidence.")
    else:
        print("Prediction confident, no review needed.")
