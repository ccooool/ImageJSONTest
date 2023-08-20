from PIL import Image
import json

def image_to_json(image_path):
    image = Image.open(image_path)
    width, height = image.size
    pixel_data = list(image.getdata())

    rgb_values = [list(pixel[:3]) for pixel in pixel_data]
    
    json_data = json.dumps(rgb_values)
    
    with open('image_data.json', 'w') as json_file:
        json_file.write(json_data)

if __name__ == "__main__":
    input_image_path = "input_image.jpg"  # Change this to your input image path
    image_to_json(input_image_path)