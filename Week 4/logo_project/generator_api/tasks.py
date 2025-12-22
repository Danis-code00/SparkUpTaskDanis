import cv2
import numpy as np
import os
import random
from pathlib import Path
from django.conf import settings
from background_task import background

@background(schedule=0)
def generate_logo_task(num_images, subset="train"):
    num_images = int(num_images)
    # 1. Setup Paths using Django Settings
    LOGO_DIR = os.path.join(settings.BASE_DIR, "logos")
    BG_DIR = os.path.join(settings.BASE_DIR, "backgrounds")
    
    img_out = Path(settings.BASE_DIR) / "dataset" / subset / "images"
    lbl_out = Path(settings.BASE_DIR) / "dataset" / subset / "labels"

    # Create directories
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    logo_files = sorted([f for f in os.listdir(LOGO_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    bg_files = [f for f in os.listdir(BG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Starting generation of {num_images} images...")

    for i in range(num_images):
        bg_path = os.path.join(BG_DIR, random.choice(bg_files))
        class_id = random.randint(0, len(logo_files) - 1)
        logo_path = os.path.join(LOGO_DIR, logo_files[class_id])

        bg = cv2.imread(bg_path)
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        
        if bg is None or logo is None: continue

        # Standard YOLO resizing
        bg = cv2.resize(bg, (640, 640))
        bg_h, bg_w = bg.shape[:2]

        # Scale and Overlay logic
        scale = random.uniform(0.1, 0.3)
        new_w = int(bg_w * scale)
        new_h = int(logo.shape[0] * (new_w / logo.shape[1]))
        logo_resized = cv2.resize(logo, (new_w, new_h))

        x_offset = random.randint(0, bg_w - new_w)
        y_offset = random.randint(0, bg_h - new_h)

        # Alpha blending for PNGs
        if logo_resized.shape[2] == 4:
            alpha_logo = logo_resized[:, :, 3] / 255.0
            alpha_bg = 1.0 - alpha_logo
            for c in range(0, 3):
                bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = (
                    alpha_logo * logo_resized[:, :, c] +
                    alpha_bg * bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
                )
        else:
            bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = logo_resized[:, :, :3]

        # Calculate YOLO Coordinates
        x_center = (x_offset + new_w / 2) / bg_w
        y_center = (y_offset + new_h / 2) / bg_h
        norm_w = new_w / bg_w
        norm_h = new_h / bg_h

        # Save results
        file_name = f"gen_{i}_{class_id}.jpg"
        cv2.imwrite(str(img_out / file_name), bg)
        with open(lbl_out / file_name.replace(".jpg", ".txt"), "w") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        if i % 10 == 0:
            print(f"Progress: {i}/{num_images} images created.")

    print("✅ Generation complete!")

