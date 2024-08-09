from flask import request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from app import app
from app.utils import load_image, save_segmentation, save_overlay_image
from app.depth_map import generate_depth_map
from app.segmentation import device, model, feature_extractor, process_and_predict
from app.evaluation import calculate_metrics

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

    if 'ground_truth' not in request.files:
        return jsonify({'error': 'No ground truth part in the request'}), 400
    
    ground_truth_file = request.files['ground_truth']
    if ground_truth_file.filename == '':
        return jsonify({'error': 'No ground truth file selected for uploading'}), 400

    ground_truth_filename = secure_filename(ground_truth_file.filename)
    ground_truth_path = os.path.join('/tmp', ground_truth_filename)
    ground_truth_file.save(ground_truth_path)

    try:
        output_dir = '/tmp'
        depth_map_path = generate_depth_map(input_path, output_dir)
        if not depth_map_path or not os.path.exists(depth_map_path):
            return jsonify({'error': 'Depth map generation failed'}), 500

        binary_mask, preprocessed_image_path = process_and_predict(depth_map_path, model, feature_extractor, device)
        
        output_seg_path = "/tmp/output_seg.png"
        output_overlay_path = "/tmp/output_overlay.png"
        
        save_segmentation(binary_mask, output_seg_path)
        original_image = load_image(depth_map_path)
        save_overlay_image(original_image, binary_mask, output_overlay_path)

        iou, dice = calculate_metrics(ground_truth_path, binary_mask)
        
        return jsonify({
            'segmentation_result': output_seg_path,
            'overlay_result': output_overlay_path,
            'iou': iou,
            'dice': dice,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory('/tmp', filename)