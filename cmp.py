import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to load and process images
def load_and_process_images(original_path, predicted_path, ground_truth_path):
    original_image = np.array(Image.open(original_path))
    predicted_mask = np.array(Image.open(predicted_path).convert("L"))
    ground_truth = np.array(Image.open(ground_truth_path).convert("L"))

    predicted_mask = (predicted_mask > 128).astype(np.uint8)
    ground_truth = (ground_truth > 128).astype(np.uint8)

    return original_image, predicted_mask, ground_truth

# Function to calculate IoU and Dice Coefficient
def calculate_metrics(ground_truth, predicted_mask):
    def IoU_metric(y_true, y_pred):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        smooth = 1e-4
        iou = (intersection + smooth) / (union + smooth)
        return iou

    def Dice_coeff(y_true, y_pred):
        smooth = 1e-4
        numerator = 2 * np.sum(y_true * y_pred)
        denominator = np.sum(y_true + y_pred)
        dice = (numerator + smooth) / (denominator + smooth)
        return dice

    mean_iou = IoU_metric(ground_truth, predicted_mask)
    mean_dice = Dice_coeff(ground_truth, predicted_mask)

    return mean_iou, mean_dice

# Function to visualize results
def visualize_results(original_image, ground_truth, predicted_mask, mean_iou, mean_dice):
    overlay = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)

    overlay[(predicted_mask == 1) & (ground_truth == 1)] = [0, 255, 0]
    overlay[(predicted_mask == 0) & (ground_truth == 1)] = [255, 0, 0]
    overlay[(predicted_mask == 1) & (ground_truth == 0)] = [0, 0, 255]

    overlayed_image = original_image.copy()
    mask_indices = np.where(overlay != 0)
    overlayed_image[mask_indices[0], mask_indices[1], :] = overlay[mask_indices[0], mask_indices[1], :]

    fig, ax = plt.subplots(1, 4, figsize=(25, 20))
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[1].imshow(ground_truth, cmap='gray')
    ax[1].set_title('Ground Truth Mask')
    ax[2].imshow(predicted_mask, cmap='gray')
    ax[2].set_title('Predicted Mask')
    ax[3].imshow(overlayed_image)
    ax[3].set_title('Overlayed Image\nGreen: Correct, Red: FN, Blue: FP')

    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    metrics = ['IoU', 'Dice Coefficient']
    values = [mean_iou, mean_dice]
    ax_bar.bar(metrics, values, color=['blue', 'green'])
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel('Value')
    ax_bar.set_title('IoU and Dice Coefficient')

    fig.text(0.5, 0.05, f'IoU: {mean_iou:.4f}', ha='center', fontsize=12)
    fig.text(0.5, 0.01, f'Dice Coefficient: {mean_dice:.4f}', ha='center', fontsize=12)

    plt.show()

# Function to compare models
def compare_models(model_results):
    fig_bar, ax_bar = plt.subplots(figsize=(15, 8))
    metrics = ['IoU', 'Dice Coefficient']

    for model_name, metrics_values in model_results.items():
        ax_bar.bar([f'{model_name} - {metric}' for metric in metrics], metrics_values, alpha=0.7, label=model_name)

    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel('Value')
    ax_bar.set_title('Comparison of IoU and Dice Coefficient Across Models')
    ax_bar.legend()

    plt.show()

# Paths for the images
original_path = "/Users/sarahbanat/Desktop/seg/data/test/test1.jpg"
predicted_path = "/Users/sarahbanat/Desktop/docker_tmp/output_seg.png"
ground_truth_path = "/Users/sarahbanat/Desktop/seg/data/test/gt.png"

# Load and process images
original_image, predicted_mask, ground_truth = load_and_process_images(original_path, predicted_path, ground_truth_path)

# Calculate metrics
mean_iou, mean_dice = calculate_metrics(ground_truth, predicted_mask)

# Visualize results
visualize_results(original_image, ground_truth, predicted_mask, mean_iou, mean_dice)

# Store results in a dictionary
model_results = {
    'Segformer': [mean_iou, mean_dice],
    # Add more models and their metrics here
}

# Compare models
compare_models(model_results)