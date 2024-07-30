import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import argparse

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def process_and_predict(image_path, model, feature_extractor, device):
    image = load_image(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, 
            size=image.size[::-1], 
            mode='bilinear', 
            align_corners=False
        )

    predicted_class = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    # Focus on utility poles (class index 5 for Cityscapes)
    utility_pole_class_index = 5
    binary_mask = np.where(predicted_class == utility_pole_class_index, 1, 0)

    return binary_mask

def save_segmentation(binary_mask, output_path):
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

def overlay_segmentation_on_image(image, binary_mask):
    overlay = Image.fromarray((binary_mask * 255).astype(np.uint8))
    overlay = overlay.convert("RGBA")

    image = image.convert("RGBA")
    overlayed_image = Image.blend(image, overlay, alpha=0.5)

    return overlayed_image

def save_overlay_image(image, binary_mask, output_path):
    overlayed_image = overlay_segmentation_on_image(image, binary_mask)
    overlayed_image.save(output_path)

def main(image_path, model_name, device, output_seg_path, output_overlay_path):
    device = torch.device(device)
    feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    binary_mask = process_and_predict(image_path, model, feature_extractor, device)
    
    # Save binary mask result
    save_segmentation(binary_mask, output_seg_path)
    
    # Save overlay image
    original_image = load_image(image_path)
    save_overlay_image(original_image, binary_mask, output_overlay_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024", help="Name of the pretrained model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the inference on.")
    parser.add_argument("--output_seg_path", type=str, required=True, help="Path to save the segmentation result image.")
    parser.add_argument("--output_overlay_path", type=str, required=True, help="Path to save the overlay image.")
    args = parser.parse_args()
    main(args.image_path, args.model_name, args.device, args.output_seg_path, args.output_overlay_path)