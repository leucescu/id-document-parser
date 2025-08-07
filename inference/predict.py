####PLACEHOLDER###

# Loads model + processor and performs prediction on new imagesfrom transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("your/saved-model")
model = VisionEncoderDecoderModel.from_pretrained("your/saved-model")

# Inference
from PIL import Image
import torch

image = Image.open("path/to/id.png").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values

outputs = model.generate(pixel_values)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print(generated_text)