import numpy as np
from scipy.signal import convolve2d
from skimage import io, color

def load_image(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return (img * 255).astype(np.uint8)

def edge_detection(image):
    if image.ndim > 2:
        image = image[:, :, 0]
        
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gx = convolve2d(image, kernel_x, mode='same', boundary='symm')
    gy = convolve2d(image, kernel_y, mode='same', boundary='symm')

    edge_magnitude = np.sqrt(gx**2 + gy**2)
    
    if edge_magnitude.max() > 0:
        edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255
        
    return edge_magnitude.astype(np.uint8)
