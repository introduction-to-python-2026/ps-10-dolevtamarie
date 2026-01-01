from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # טעינה והמרה לגווני אפור (L)
    img = Image.open(path).convert('L')
    # המרה למערך נומפי
    arr = np.array(img)
    # וידוא שהמערך הוא דו-ממדי בלבד (מוריד ממדים ריקים אם יש)
    if arr.ndim > 2:
        arr = arr[:, :, 0]
    return arr

def edge_detection(image):
    # הגדרת קרנל Sobel
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # ביצוע קונבולוציה
    gx = convolve2d(image, kx, mode='same', boundary='symm')
    gy = convolve2d(image, ky, mode='same', boundary='symm')
    
    # חישוב עוצמת הקצה (Magnitude)
    edge = np.sqrt(gx**2 + gy**2)
    
    return edge
