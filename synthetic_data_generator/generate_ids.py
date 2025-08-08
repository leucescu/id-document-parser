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
        self.training_data_size = 4000
        self.validation_data_size = 1000

        # Number of images that will be used for generating background
        self.background_data_size = 200
        self.face_data_size = 500

    def create_background_images(self):
        """Download and save background images from the DTD dataset."""
        try:
            # Load dataset from HuggingFace
            print("Loading dataset...")
            dtd = load_dataset(
                "dream-textures/textures-color-1k", 
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
    def paste_id_on_background_with_padding(background, id_card):
        # Ensure both are RGBA
        if id_card.mode != 'RGBA':
            id_card = id_card.convert('RGBA')
        if background.mode != 'RGBA':
            background = background.convert('RGBA')

        bg_w, bg_h = background.size
        id_w, id_h = id_card.size

        # Calculate new size to fully contain both
        new_w = max(bg_w, id_w)
        new_h = max(bg_h, id_h)

        # If background is smaller than new size, resize/stretch it to fill new size
        if bg_w != new_w or bg_h != new_h:
            background = background.resize((new_w, new_h), resample=Image.BICUBIC)

        # Create a new blank RGBA canvas
        new_bg = Image.new('RGBA', (new_w, new_h), (255, 255, 255, 255))

        # Paste resized background at (0,0) — fills whole new canvas
        new_bg.paste(background, (0, 0))

        # Paste ID card centered over the new background, preserving transparency mask
        id_x = (new_w - id_w) // 2
        id_y = (new_h - id_h) // 2
        new_bg.paste(id_card, (id_x, id_y), id_card)

        return new_bg

    @staticmethod
    def apply_perspective_transform(pil_img):
        pil_img = pil_img.convert("RGBA")
        w, h = pil_img.size

        max_shift = int(min(w, h) * 0.1)

        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        dst = src + np.random.uniform(-max_shift, max_shift, size=src.shape).astype(np.float32)

        return Synthetic_Data.perspective_transform_rgba(pil_img, src, dst)


    @staticmethod
    def apply_color_augmentations(pil_img):
        pil_img = pil_img.convert("RGBA")
        img_np = np.array(pil_img.convert("RGB"))

        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.GaussNoise(var_limit=(5, 20), p=0.15),
        ])

        augmented = transform(image=img_np)["image"]

        alpha_channel = pil_img.split()[-1]
        augmented_img = Image.fromarray(augmented).convert("RGBA")
        augmented_img.putalpha(alpha_channel)

        return augmented_img

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
        
        
        # Create a copy of background to paste on
        composite = background.copy()
        
        composite = Synthetic_Data.paste_id_on_background_with_padding(background, rotated_id)

        # # Optionally resize final_composite back to (512, 256)
        # final_resized = composite.resize((512, 256), resample=Image.BICUBIC)
    
        return composite.convert('RGB')  # Convert back to RGB
    
    @staticmethod
    def perspective_transform_rgba(pil_img, src_points, dst_points):
        """
        Apply perspective transform on RGBA image without cropping by calculating
        the bounding box of the warped image and translating accordingly.

        Args:
            pil_img (PIL.Image): Input RGBA image.
            src_points (np.array): Source points for perspective transform (4 points).
            dst_points (np.array): Destination points (4 points).

        Returns:
            PIL.Image: Transformed image with transparency preserved.
        """

        img_np = np.array(pil_img)
        h, w = img_np.shape[:2]

        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))

        # Transform the corners to find bounding box
        corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32).reshape(-1, 1, 2)

        warped_corners = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

        # Find bounding box of warped corners
        min_x = np.min(warped_corners[:, 0])
        min_y = np.min(warped_corners[:, 1])
        max_x = np.max(warped_corners[:, 0])
        max_y = np.max(warped_corners[:, 1])

        new_w = int(np.ceil(max_x - min_x))
        new_h = int(np.ceil(max_y - min_y))

        # Translation matrix to shift image so all pixels are positive
        translation = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ])

        # Combine translation with perspective transform
        M_translated = translation @ M

        # Warp image with combined matrix and new size
        warped = cv2.warpPerspective(img_np, M_translated, (new_w, new_h),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))  # transparent background

        return Image.fromarray(warped)

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

        # Apply perspective transform first
        perspective_img = Synthetic_Data.apply_perspective_transform(textured_id)

        # For demo: pick a random rotation angle, e.g. -10 to +10 degrees
        angle = random.uniform(-30, 30)

        # Load a random background and resize
        bg_files = [f for f in os.listdir(self.save_bck_dir) if f.lower().endswith(('.jpg', '.png'))]
        bg_path = os.path.join(self.save_bck_dir, random.choice(bg_files))
        background = Image.open(bg_path).convert("RGBA").resize((512, 256))

        # Rotate the augmented ID card and composite it over background
        rotated_image = Synthetic_Data.rotate_id_over_background(background, perspective_img, angle)

        # Then apply color augmentations on the warped image
        final_image = Synthetic_Data.apply_color_augmentations(rotated_image)

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
    synthetic_data.create_background_images()
    synthetic_data.create_synthetic_face_images()
    synthetic_data.create_training_data()
    synthetic_data.create_validation_data()


if __name__ == "__main__":
    main()
