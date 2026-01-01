from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # טעינה והמרה פשוטה. בלי שטיקים.
    return np.array(Image.open(path).convert('L'))

def edge_detection(image):
    # הגדרת הקרנלים (בלי dtype, בלי כלום)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # קונבולוציה פשוטה
    gx = convolve2d(image, kx, mode='same', boundary='symm')
    gy = convolve2d(image, ky, mode='same', boundary='symm')

    # חישוב עוצמה
    return np.sqrt(gx**2 + gy**2)
