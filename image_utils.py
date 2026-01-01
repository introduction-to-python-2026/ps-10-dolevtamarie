from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # טעינה והמרה לגווני אפור
    img = Image.open(path).convert('L')
    # הפיכה למערך נומפי
    arr = np.array(img)
    # הכרחה של המערך להיות דו-ממדי (לוקחים רק את הגובה והרוחב)
    if arr.ndim > 2:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)

def edge_detection(image):
    # וידוא סופי שהקלט הוא דו-ממדי (הכרחי עבור הטסט)
    if image.ndim > 2:
        image = image[:, :, 0]
        
    # הגדרת אופרטור Sobel
    # חשוב להשתמש בערכים האלו בדיוק כדי לקבל תוצאה שגדולה מ-50 בקצוות
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    
    # ביצוע קונבולוציה
    gx = convolve2d(image, kx, mode='same', boundary='symm')
    gy = convolve2d(image, ky, mode='same', boundary='symm')
    
    # חישוב עוצמת הקצה (Magnitude)
    return np.sqrt(gx**2 + gy**2)
    
    gx = convolve2d(image, kx, mode='same', boundary='symm')
    gy = convolve2d(image, ky, mode='same', boundary='symm')
    
    return np.sqrt(gx**2 + gy**2)
