from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # טעינה עם PIL והמרה לשחור-לבן
    img = Image.open(path).convert('L')
    # המרה למערך נומפי
    arr = np.array(img)
    # אם המערך הגיע עם ממד נוסף (למשל 1 בקצה), אנחנו מסירים אותו
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], arr.shape[1])
    return arr

def edge_detection(image):
    # וידוא נוסף שהתמונה דו-ממדית לפני החישוב
    if image.ndim > 2:
        image = image.reshape(image.shape[0], image.shape[1])
        
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    gx = convolve2d(image, kx, mode='same', boundary='symm')
    gy = convolve2d(image, ky, mode='same', boundary='symm')
    
    return np.sqrt(gx**2 + gy**2)
