import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import color, exposure
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the uploaded image for the model.
    
    Args:
        image: PIL Image object
        target_size: Tuple of (width, height) for resizing
        
    Returns:
        Preprocessed image as numpy array
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if it's grayscale
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize image
    img_resized = cv2.resize(img_array, target_size)
    
    return img_resized

def analyze_texture(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Analyze texture features using GLCM (Gray-Level Co-occurrence Matrix).
    
    Args:
        image: PIL Image object
        distances: List of pixel pair distance offsets
        angles: List of pixel pair angles in radians
        
    Returns:
        Dictionary containing texture features
    """
    # Convert to grayscale if not already
    img_array = np.array(image.convert('L'))
    
    # Normalize the image
    img_array = img_array.astype(np.uint8)
    
    # Calculate GLCM matrix
    glcm = graycomatrix(img_array, distances=distances, angles=angles, 
                        symmetric=True, normed=True)
    
    # Calculate GLCM properties
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    # Return texture features
    return {
        'contrast': float(contrast),
        'dissimilarity': float(dissimilarity),
        'homogeneity': float(homogeneity),
        'energy': float(energy),
        'correlation': float(correlation)
    }

def analyze_color_profile(image):
    """
    Analyze color profile of the image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary containing color profile metrics
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if it's not
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Convert to different color spaces
    img_hsv = color.rgb2hsv(img_array)
    img_lab = color.rgb2lab(img_array)
    
    # Calculate color statistics
    # RGB
    r_mean, g_mean, b_mean = np.mean(img_array[:,:,0]), np.mean(img_array[:,:,1]), np.mean(img_array[:,:,2])
    r_std, g_std, b_std = np.std(img_array[:,:,0]), np.std(img_array[:,:,1]), np.std(img_array[:,:,2])
    
    # HSV
    h_mean, s_mean, v_mean = np.mean(img_hsv[:,:,0]), np.mean(img_hsv[:,:,1]), np.mean(img_hsv[:,:,2])
    h_std, s_std, v_std = np.std(img_hsv[:,:,0]), np.std(img_hsv[:,:,1]), np.std(img_hsv[:,:,2])
    
    # LAB
    l_mean, a_mean, b_mean_lab = np.mean(img_lab[:,:,0]), np.mean(img_lab[:,:,1]), np.mean(img_lab[:,:,2])
    l_std, a_std, b_std_lab = np.std(img_lab[:,:,0]), np.std(img_lab[:,:,1]), np.std(img_lab[:,:,2])
    
    return {
        'rgb': {
            'r_mean': float(r_mean), 'g_mean': float(g_mean), 'b_mean': float(b_mean),
            'r_std': float(r_std), 'g_std': float(g_std), 'b_std': float(b_std)
        },
        'hsv': {
            'h_mean': float(h_mean), 's_mean': float(s_mean), 'v_mean': float(v_mean),
            'h_std': float(h_std), 's_std': float(s_std), 'v_std': float(v_std)
        },
        'lab': {
            'l_mean': float(l_mean), 'a_mean': float(a_mean), 'b_mean': float(b_mean_lab),
            'l_std': float(l_std), 'a_std': float(a_std), 'b_std': float(b_std_lab)
        }
    }

def extract_roi(image, x, y, width, height):
    """
    Extract Region of Interest from the image.
    
    Args:
        image: PIL Image object
        x, y: Top-left corner coordinates
        width, height: Dimensions of ROI
        
    Returns:
        ROI as PIL Image
    """
    return image.crop((x, y, x + width, y + height))

def plot_color_distribution(image):
    """
    Generate a plot of color distribution histograms.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string of the plot image
    """
    img_array = np.array(image)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot RGB histograms
    color_channels = ['Red', 'Green', 'Blue']
    colors = ['r', 'g', 'b']
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_array], [i], None, [256], [0, 256])
        ax[i].plot(hist, color=color)
        ax[i].set_xlim([0, 256])
        ax[i].set_title(f'{color_channels[i]} Channel')
        ax[i].set_xlabel('Pixel Value')
        ax[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode image to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_str
