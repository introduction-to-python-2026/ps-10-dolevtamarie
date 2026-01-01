from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # טעינה והמרה לגווני אפור
    img = Image.open(path).convert('L')
    # הפיכה למערך נומפי
    arr = np.array(img)
    # הכרחה של המערך להיות דו-ממדי בלבד
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], arr.shape[1])
    return arr

def edge_detection(image):
    # יצירת הקרנלים של Sobel
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    
    # וידוא שהקלט ל-convolve2d הוא דו-ממדי
    if image.ndim != 2:
        image = image.reshape(image.shape[0], image.shape[1])
        
    # ביצוע קונבולוציה
    gx = convolve2d(image, kx, mode='same', boundary='symm')
    gy = convolve2d(image, ky, mode='same', boundary='symm')
    
    # חישוב עוצמת הקצה
    return np.sqrt(gx**2 + gy**2)
    
    gx = convolve2d(image, kx, mode='same', boundary='symm')
    gy = convolve2d(image, ky, mode='same', boundary='symm')
    
    return np.sqrt(gx**2 + gy**2)
