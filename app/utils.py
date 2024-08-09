import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import imgaug.augmenters as iaa

def kmeans_clustering(image, n_clusters=2):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    pixels = image.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pixels)
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    
    clustered_image = clustered.reshape(image.shape).astype(np.uint8)
    
    clustered_image = cv2.cvtColor(clustered_image, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(clustered_image)

def augment_image(image):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.5),  # vertical flips
        iaa.Affine(rotate=(-20, 20)),  # random rotations
        iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),  # add gaussian noise
    ])
    
    image_aug = seq(image=np.array(image))
    
    return Image.fromarray(image_aug)

def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    #  K-Means clustering
    #clustered_image = kmeans_clustering(image, n_clusters=2)
    
    #  image augmentation
    #augmented_image = augment_image(image)
    
    return image #return normal image with no processing (after testing results are significantly worse, so for now no preprocessing will be applied)

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def save_segmentation(binary_mask, output_path):
    plt.imsave(output_path, binary_mask, cmap='gray')

def overlay_segmentation_on_image(image, binary_mask):
    overlay = Image.fromarray((binary_mask * 255).astype(np.uint8)).convert("RGBA")
    image = image.convert("RGBA")
    return Image.blend(image, overlay, alpha=0.5)

def save_overlay_image(image, binary_mask, output_path):
    overlayed_image = overlay_segmentation_on_image(image, binary_mask)
    overlayed_image.save(output_path)