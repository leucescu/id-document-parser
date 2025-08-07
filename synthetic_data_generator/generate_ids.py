from datasets import load_dataset
from faker import Faker
# Python Imaging Library
from PIL import Image, ImageDraw, ImageFont
import os
import json

import albumentations as A
import cv2
import numpy as np
import random
from PIL import Image

FONT_PATH = "C:\\Windows\\Fonts\\Arial.ttf"  # Change this based on OS

class Synthetic_Data:
    def __init__(self):
        self.faker = Faker()

        self.font_path = FONT_PATH
        self.save_train_dir = "data/train"
        self.save_val_dir = "data/validation"
        self.save_bck_dir = "data/background"
        self.texture_path = "data/card_textures"
        self.save_face_dir = "data/faces"

        # Size defined as number of generated images
        self.training_data_size = 10
        self.validation_data_size = 10

        # Number of images that will be used for generating background
        self.background_data_size = 50
        self.face_data_size = 150

    def create_background_images(self):
        """Download and save background images from the DTD dataset."""
        try:
            # Load dataset from HuggingFace
            print("Loading dataset...")
            dtd = load_dataset(
                "cansa/Describable-Textures-Dataset-DTD", 
                split="train"
            )

            # Ensure the output directory exists
            os.makedirs(self.save_bck_dir, exist_ok=True)

            # Save images (limited by background_data_size)
            for i, example in enumerate(dtd.select(range(self.background_data_size))):
                img = example["image"]
                img.save(os.path.join(self.save_bck_dir, f"bg_{i:03d}.jpg"))  # Use os.path.join for cross-platform paths

            print(f"Successfully saved {self.background_data_size} background images to {self.save_bck_dir}")

        except Exception as e:
            print(f"Error: {e}")
            raise

    def create_synthetic_face_images(self):
        """Download and save synthetic face images from a Hugging Face dataset."""
        try:
            print("Loading synthetic face dataset...")
            
            # Replace 'generated-photos/faces' with the actual dataset name you want to use
            faces_ds = load_dataset(
                "TLeonidas/this-person-does-not-exist",
                split="train"
            )
            
            # Ensure output directory exists
            os.makedirs(self.save_face_dir, exist_ok=True)
            
            # Save images (limit by face_data_size)
            for i, example in enumerate(faces_ds.select(range(self.face_data_size))):
                img = example["image"]  # Assuming the image is stored in the "image" key
                
                # Save the image
                img.save(os.path.join(self.save_face_dir, f"face_{i:03d}.jpg"))
            
            print(f"Successfully saved {self.face_data_size} face images to {self.save_face_dir}")

        except Exception as e:
            print(f"Error: {e}")
            raise

    # @staticmethod
    # def add_background_with_mask(id_img, backgrounds_folder):
    #     # Load random background
    #     bg_files = [f for f in os.listdir(backgrounds_folder) if f.lower().endswith(('.jpg', '.png'))]
    #     bg_path = os.path.join(backgrounds_folder, random.choice(bg_files))
    #     background = Image.open(bg_path).convert("RGB")

    #     # Resize background to fixed size (you can make this a class variable if needed)
    #     bg = background.resize((512, 256))
    #     id_img = id_img.convert("RGB")  # Just to be sure

    #     # Resize ID to be slightly smaller than background
    #     id_img = id_img.resize((400, 200))
    #     id_np = np.array(id_img)
    #     bg_np = np.array(bg)

    #     # Center position
    #     bg_w, bg_h = bg.size
    #     id_w, id_h = id_img.size
    #     x = (bg_w - id_w) // 2
    #     y = (bg_h - id_h) // 2

    #     # Extract region of interest from background
    #     roi = bg_np[y:y+id_h, x:x+id_w]

    #     # White mask (all-white pixels)
    #     white_mask = np.all(id_np >= 255, axis=-1)

    #     # Composite: keep background in white areas, overlay text from ID
    #     roi[~white_mask] = id_np[~white_mask]

    #     # Put ROI back into background
    #     bg_np[y:y+id_h, x:x+id_w] = roi

    #     return Image.fromarray(bg_np)

    def draw_face_on_id(self, id_img, face_size=(120, 120), position=None):
        """
        Draw a random face image from the saved face directory onto the given ID card image.

        Args:
            id_img (PIL.Image): The ID card image to draw on.
            face_size (tuple): Desired size (width, height) to resize the face image.
            position (tuple or None): (x, y) position to paste the face image on the ID card.
                                    If None, defaults to top-right with 20 px padding.

        Returns:
            PIL.Image: ID card image with the face pasted on it.
        """
        # List face image files
        face_files = [f for f in os.listdir(self.save_face_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not face_files:
            print("No face images found in", self.save_face_dir)
            return id_img  # Return original if no face available

        # Pick a random face image
        face_path = os.path.join(self.save_face_dir, random.choice(face_files))
        face_img = Image.open(face_path).convert("RGB")

        # Resize face
        face_img = face_img.resize(face_size)

        # Default position: top-right corner with 20 px padding
        if position is None:
            x = id_img.width - face_size[0] - 20
            y = 20
        else:
            x, y = position

        # Paste face onto the ID card image
        id_img.paste(face_img, (x, y))

        return id_img

    @staticmethod
    def apply_augmentations(pil_img):
        """ Method used to make position and orientation of an ID card more realistic by applying space transformations."""

        img_np = np.array(pil_img)
        transform = A.Compose([
            # A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
            A.RandomBrightnessContrast(p=0.5),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Perspective(scale=(0.05, 0.1), p=0.7),
        ])
        augmented = transform(image=img_np)["image"]
        return Image.fromarray(augmented)

    @staticmethod
    def apply_texture_base(image, texture_folder):
        """ Method used to apply a texture to the ID card."""

        # Load random texture
        texture_files = [f for f in os.listdir(texture_folder) if f.endswith(('.jpg', '.png'))]
        texture_path = os.path.join(texture_folder, random.choice(texture_files))
        texture = Image.open(texture_path).convert("RGB")

        # Resize/crop texture to match card size
        texture = texture.resize(image.size)

        # Blend texture into white base with some opacity
        texture = texture.convert("RGBA")
        texture.putalpha(80)  # Lower value = more transparent

        base = image.convert("RGBA")
        textured_card = Image.alpha_composite(base, texture)

        return textured_card.convert("RGB")

    @staticmethod
    def rotate_id_over_background(background, id_card, angle):
        # Ensure id_card has alpha channel
        if id_card.mode != 'RGBA':
            id_card = id_card.convert('RGBA')
        if background.mode != 'RGBA':
            background = background.convert('RGBA')
        
        # Rotate id_card with transparent fill, expand canvas
        rotated_id = id_card.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0,0,0,0))
        
        # Compute position to paste rotated ID onto background (centered)
        bg_w, bg_h = background.size
        id_w, id_h = rotated_id.size
        
        x = (bg_w - id_w) // 2
        y = (bg_h - id_h) // 2
        
        # Create a copy of background to paste on
        composite = background.copy()
        
        # Paste rotated id_card on background using alpha channel as mask
        composite.paste(rotated_id, (x, y), rotated_id)
    
        return composite.convert('RGB')  # Convert back to RGB

    @staticmethod
    def expand_canvas(image, padding=100, background=(255, 255, 255)):
        w, h = image.size
        new_w, new_h = w + 2 * padding, h + 2 * padding
        new_image = Image.new("RGB", (new_w, new_h), background)
        new_image.paste(image, (padding, padding))
        return new_image

    def generate_fake_id(self, save_dir, idx=0):
        os.makedirs(save_dir, exist_ok=True)

        # Generate fake data
        name = self.faker.name()
        dob = self.faker.date_of_birth(minimum_age=18, maximum_age=90).strftime("%Y-%m-%d")
        id_number = self.faker.unique.random_int(min=100000000, max=999999999)
        expiry = self.faker.future_date().strftime("%Y-%m-%d")

        label = {
            "name": name,
            "dob": dob,
            "id_number": str(id_number),
            "expiry": expiry
        }

        # Create blank transparent image for ID card (RGBA)
        id_card = Image.new("RGBA", (400, 200), (255, 255, 255, 0))  # transparent
        
        # Apply texture base on id_card (convert back to RGBA)
        textured_id = Synthetic_Data.apply_texture_base(id_card.convert("RGB"), self.texture_path).convert("RGBA")
        
        # Draw face and text on textured_id (convert to RGBA so paste works)
        textured_id = self.draw_face_on_id(textured_id.convert("RGBA"))
        draw = ImageDraw.Draw(textured_id)

        try:
            font = ImageFont.truetype(self.font_path, 20)
        except IOError:
            font = ImageFont.load_default()

        draw.text((20, 20), f"Name: {name}", fill="black", font=font)
        draw.text((20, 60), f"DOB: {dob}", fill="black", font=font)
        draw.text((20, 100), f"ID: {id_number}", fill="black", font=font)
        draw.text((20, 140), f"Expiry: {expiry}", fill="black", font=font)

        # id_card = Synthetic_Data.expand_canvas(textured_id, padding=400, background=(255,255,255,0))  # transparent background
        augmented_id = Synthetic_Data.apply_augmentations(textured_id.convert("RGB")).convert("RGBA")

        # For demo: pick a random rotation angle, e.g. -10 to +10 degrees
        angle = random.uniform(-30, 30)

        # Load a random background and resize
        bg_files = [f for f in os.listdir(self.save_bck_dir) if f.lower().endswith(('.jpg', '.png'))]
        bg_path = os.path.join(self.save_bck_dir, random.choice(bg_files))
        background = Image.open(bg_path).convert("RGBA").resize((512, 256))

        # Rotate the augmented ID card and composite it over background
        final_image = Synthetic_Data.rotate_id_over_background(background, augmented_id, angle)

        # Save image and label
        img_path = os.path.join(save_dir, f"id_{idx:04d}.png")
        label_path = os.path.join(save_dir, f"id_{idx:04d}.json")

        final_image.save(img_path)
        with open(label_path, "w") as f:
            json.dump(label, f)

        print(f"Saved {img_path}")

    def create_training_data(self):
        for i in range(self.training_data_size):
            self.generate_fake_id(self.save_train_dir, idx=i)

    def create_validation_data(self):
        for i in range(self.validation_data_size):
            self.generate_fake_id(self.save_val_dir, idx=i)


def main ():
    synthetic_data = Synthetic_Data()
    # synthetic_data.create_background_images()
    # synthetic_data.create_synthetic_face_images()
    synthetic_data.create_training_data()
    synthetic_data.create_validation_data()


if __name__ == "__main__":
    main()
