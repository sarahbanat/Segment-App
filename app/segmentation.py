import torch
from PIL import Image
import numpy as np
import os
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from app.utils import preprocess_image  

def initialize_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    return device, model, feature_extractor

# initialize the model
device, model, feature_extractor = initialize_model("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

def process_and_predict(image_path, model, feature_extractor, device):
    #preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    preprocessed_image_path = os.path.join('/tmp', 'preprocessed_image.png')
    preprocessed_image.save(preprocessed_image_path)

    # convert preprocessed image to model input format
    inputs = feature_extractor(images=preprocessed_image, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, 
            size=preprocessed_image.size[::-1], 
            mode='bilinear', 
            align_corners=False
        )

    predicted_class = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    utility_pole_class_index = 5
    binary_mask = np.where(predicted_class == utility_pole_class_index, 1, 0)
    
    return binary_mask, preprocessed_image_path