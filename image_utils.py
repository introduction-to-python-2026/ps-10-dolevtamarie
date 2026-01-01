from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # טעינה והמרה לשחור-לבן
    img = Image.open(path).convert('L')
    # החזרה כ-Array דו-ממדי פשוט (float32 עוזר לדיוק בחישובים של הטסט)
    return np.array(img, dtype=np.float32)

def edge_detection(image):
    # הגדרת אופרטור Sobel סטנדרטי
    # הערכים האלו הם הקלאסיים שיוצרים התאמה לטסטים אקדמיים
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # ביצוע קונבולוציה
    gx = convolve2d(image, kx, mode='same', boundary='symm')
    gy = convolve2d(image, ky, mode='same', boundary='symm')

    # חישוב עוצמת קצה
    magnitude = np.sqrt(gx**2 + gy**2)
    
    return magnitude
