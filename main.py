from image_utils import load_image, edge_detection
from PIL import Image

def main():
    # השתמשי בתמונה פשוטה, אולי אפילו תורידי תמונה אחרת מהאינטרנט
    img = load_image("dog_pic.jpg") 
    edges = edge_detection(img)
    # שמירה בפורמט PNG (שלא מאבד נתונים כמו JPG)
    Image.fromarray(edges.astype('uint8')).save("edge_detected_result.png")

if __name__ == "__main__":
    main()
