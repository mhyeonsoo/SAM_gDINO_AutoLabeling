#!/usr/bin/env python3
import argparse
import os
import copy
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Import Grounding DINO modules
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Other modules
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as TT

# COCO Annotation modules (segmentation, bbox calculation)
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import datetime

# Set GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


########################################
# 1. Load and preprocess image
########################################
def load_image(image_path):
    """
    Load an image and apply preprocessing for Grounding DINO.
    """
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)  # 3 x H x W tensor
    return image_pil, image_tensor


########################################
# 2. Load the model
########################################
def load_model(model_config_path, model_checkpoint_path, device):
    """
    Load the Grounding DINO model using the configuration and checkpoint files.
    """
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loading result:", load_res)
    model.eval()
    return model


########################################
# 3. Convert bounding box coordinates
########################################
def convert_box(box, width, height):
    """
    Convert model output box (normalized [cx, cy, w, h]) 
    to absolute pixel coordinates [x_min, y_min, x_max, y_max].
    """
    cx, cy, w, h = box.tolist()
    x_min = int((cx - w / 2) * width)
    y_min = int((cy - h / 2) * height)
    x_max = int((cx + w / 2) * width)
    y_max = int((cy + h / 2) * height)
    return [x_min, y_min, x_max, y_max]


########################################
# 4. Run model inference
########################################
def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    """
    Run the Grounding DINO model on an image with a given caption 
    and return bounding boxes and predicted phrases.
    """
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # Apply threshold filtering
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]  # num_filt x 256
    boxes_filt = boxes[filt_mask]    # num_filt x 4

    # Extract phrases from model tokenizer
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit in logits_filt:
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


########################################
# 5. Visualization functions
########################################
def show_mask(mask, ax, random_color=True):
    """
    Display mask for debugging or visualization.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    """
    Display bounding box and label for debugging or visualization.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label, fontsize=12, color='green')


########################################
# 6. Main function
########################################
def main():
    parser = argparse.ArgumentParser(
        description="Run Grounding DINO on images in a directory and generate COCO-style JSON annotations."
    )
    parser.add_argument("target_dir", type=str, help="Directory containing image files.")
    parser.add_argument("--config_file", type=str,
                        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                        help="Model configuration file path.")
    parser.add_argument("--checkpoint", type=str,
                        default="GroundingDINO/weights/groundingdino_swinb_cogcoor.pth",
                        help="Model checkpoint file path.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu).")
    parser.add_argument("--text_prompt", type=str,
                        default="cauliflower . broccoli . zucchini",
                        help="Text prompt for object detection.")
    parser.add_argument("--box_threshold", type=float, default=0.30,
                        help="Threshold for filtering bounding boxes.")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="Threshold for filtering detected phrases.")
    parser.add_argument("--iou_threshold", type=float, default=0.8,
                        help="IoU threshold for duplicate box removal.")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.config_file, args.checkpoint, args.device)
    print("Model loaded successfully.")

    for file in os.listdir(args.target_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_path = os.path.join(args.target_dir, file)
            print("\nProcessing:", image_path)
            try:
                image_pil, image_tensor = load_image(image_path)
            except Exception as e:
                print(f"Failed to load image: {e}")
                continue
            width, height = image_pil.size

            boxes_filt, pred_phrases = get_grounding_output(
                model, image_tensor, args.text_prompt,
                args.box_threshold, args.text_threshold,
                with_logits=True, device=args.device
            )
            if boxes_filt.shape[0] == 0:
                print("No boxes detected.")
                continue

            boxes_abs = [convert_box(box, width, height) for box in boxes_filt]

            # Save JSON output
            output_json_path = os.path.join(args.target_dir, os.path.splitext(file)[0] + ".json")
            with open(output_json_path, 'w') as f:
                json.dump({"boxes": boxes_abs, "labels": pred_phrases}, f, indent=4)
            print(f"COCO annotation saved: {output_json_path}")

if __name__ == "__main__":
    main()
