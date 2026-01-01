from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # טעינה והמרה לגווני אפור
    img = Image.open(path).convert('L')
    # הפיכה למערך נומפי
    arr = np.array(img)
    # הכרחה אגרסיבית של המערך להיות דו-ממדי (ללא ערוץ צבע בכלל)
    return np.squeeze(arr).astype(np.uint8)

def edge_detection(image):
    # וידוא שהקלט הוא דו-ממדי עבור ה-convolve
    if image.ndim > 2:
        image = np.squeeze(image)
        
    # הגדרת הקרנלים של Sobel
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    
    # ביצוע קונבולוציה
    gx = convolve2d(image, kx, mode='same', boundary='symm')
    gy = convolve2d(image, ky, mode='same', boundary='symm')
    
    # חישוב עוצמה (Magnitude)
    return np.sqrt(gx**2 + gy**2)
