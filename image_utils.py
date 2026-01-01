from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    img = Image.open(path).convert('L')
    return np.array(img)

def edge_detection(image):
    if image.ndim == 3:
        image = image[:, :, 0]
        
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gx = convolve2d(image, kernel_x, mode='same', boundary='symm')
    gy = convolve2d(image, kernel_y, mode='same', boundary='symm')

    edge_magnitude = np.sqrt(gx**2 + gy**2)
    
    if edge_magnitude.max() > 0:
        edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255
        
    return edge_magnitude.astype(np.uint8)
    
    if edge_magnitude.max() > 0:
        edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255
        
    return edge_magnitude.astype(np.uint8)
    
