import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import subprocess

app = Flask(__name__)

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def generate_depth_map(image_path, output_dir):
    try:
        abs_image_path = os.path.abspath(image_path)
        abs_output_dir = os.path.abspath(output_dir)

        if not os.path.exists(abs_output_dir):
            os.makedirs(abs_output_dir)

        depth_anything_v2_dir = "/app/Depth-Anything-V2"
        print(f"Generating depth map for image: {abs_image_path}")
        print(f"Output directory: {abs_output_dir}")

        # Run depth anything v2 model to generate depth map
        cmd = [
            "python", os.path.join(depth_anything_v2_dir, "run.py"),
            "--encoder", "vitl",
            "--img-path", abs_image_path,
            "--outdir", abs_output_dir,
            "--pred-only"
        ]
        print(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, check=True, cwd=depth_anything_v2_dir, capture_output=True, text=True)

        print(f"Command output: {result.stdout}")
        print(f"Command error (if any): {result.stderr}")

        depth_map_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        depth_map_path = os.path.join(abs_output_dir, depth_map_filename)
        print(f"Expected depth map path: {depth_map_path}")
        return depth_map_path
    except subprocess.CalledProcessError as e:
        print(f"Error generating depth map for {image_path}: {e}")
        return None

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
    plt.imsave(output_path, binary_mask, cmap='gray')

def overlay_segmentation_on_image(image, binary_mask):
    overlay = Image.fromarray((binary_mask * 255).astype(np.uint8))
    overlay = overlay.convert("RGBA")

    image = image.convert("RGBA")
    overlayed_image = Image.blend(image, overlay, alpha=0.5)

    return overlayed_image

def save_overlay_image(image, binary_mask, output_path):
    overlayed_image = overlay_segmentation_on_image(image, binary_mask)
    overlayed_image.save(output_path)

@app.route('/segment', methods=['POST'])
def segment_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join('/tmp', filename)
    file.save(input_path)

    try:
        # Generate depth map
        output_dir = '/tmp'
        depth_map_path = generate_depth_map(input_path, output_dir)
        if not depth_map_path or not os.path.exists(depth_map_path):
            print(f"Depth map path does not exist: {depth_map_path}")
            return jsonify({'error': 'Depth map generation failed'}), 500

        print(f"Depth map generated at: {depth_map_path}")

        # model and device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
        
        # Process and predict on depth map
        binary_mask = process_and_predict(depth_map_path, model, feature_extractor, device)
        
        output_seg_path = "/tmp/output_seg.png"
        output_overlay_path = "/tmp/output_overlay.png"
        
    
        save_segmentation(binary_mask, output_seg_path)
        original_image = load_image(depth_map_path)
        save_overlay_image(original_image, binary_mask, output_overlay_path)
        
        return jsonify({
            'segmentation_result': output_seg_path,
            'overlay_result': output_overlay_path
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)