import numpy as np
from PIL import Image
import tensorflow as tf

def IoU_metric(y_true, y_pred):

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    
    #smooth factor to   avoid division by zero (useful in cases where the intersection or union might be very small ensuring the computation remains stable)
    smooth = 1e-4
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def Dice_coeff(y_true, y_pred):
    smooth = 1e-4
    # 2 * intersection of true and predicted masks
    numerator = 2 * np.sum(y_true * y_pred)
    
    #sum of true and predicted masks
    denominator = np.sum(y_true + y_pred)

    dice = (numerator + smooth) / (denominator + smooth)
    
    return dice

def calculate_metrics(ground_truth_path, binary_mask):

    true_mask = np.array(Image.open(ground_truth_path).convert("L")) #convert ground truth to grayscale 
    true_mask = (true_mask > 128).astype(np.uint8) #binarize gt mask : values > 128 -> set to 1, o.w : zero
    
    true_mask_resized = np.array(Image.fromarray(true_mask).resize(binary_mask.shape[::-1], Image.NEAREST))

    mean_iou = IoU_metric(true_mask_resized, binary_mask)
    mean_dice = Dice_coeff(true_mask_resized, binary_mask)
    
    return mean_iou, mean_dice