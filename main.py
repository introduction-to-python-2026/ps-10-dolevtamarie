from image_utils import load_image, edge_detection
from PIL import Image

def main():
    input_image_path = "dog_pic.jpg"
    
    img_array = load_image(input_image_path)
    
    edges = edge_detection(img_array)
    
    result_img = Image.fromarray(edges)
    result_img.save("edge_detected_result.png")

if __name__ == "__main__":
    main()
