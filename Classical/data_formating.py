import os
import json
import numpy as np
from PIL import Image, ImageDraw
from sklearn.preprocessing import LabelEncoder

directory = "data/"

def load_data(directory):
    data = []
    for name in os.listdir(directory):
        if name.endswith(".json"):
            file_path = os.path.join(directory, name)
            with open(file_path, 'r') as file:
                data.append(json.load(file))
    return data 

def vectorize(directory):
    data = load_data(directory)
    vectorized_data = []
    stroke_vectors = []
    
    for item in data:

        for label, strokes in item["drawings"].items():
            
            vector = []
            sa_vector = []
            
            for i, stroke in enumerate(strokes):
                for x,y in stroke:
                    vector.extend([x,y])
                    sa_vector.append((x, y, i))
                    
            # Flatten the vector and store it in the json_data
            if len(vector) == 0:
                continue
            
            vectorized_data.append({
                "id": item["session"],
                "label": label,
                "vector": vector,
            })

            stroke_vectors.append({
                "id": item["session"],
                "label": label,
                "sa_vector": sa_vector,
                "no_strokes": len(strokes),
                "no_points": len(vector) // 2,  
            })

    return vectorized_data, stroke_vectors

def draw_strokes_as_images(stroke_vectors, image_size=256, output_dir="Classical/Image_data"):
    os.makedirs(output_dir, exist_ok=True)

    for item in stroke_vectors:
        label = item["label"]
        sa_vector = item["sa_vector"]
        img_id = item["id"]

        # Create blank white image
        img = Image.new("L", (image_size, image_size), color=255)
        draw = ImageDraw.Draw(img)

        # Draw strokes
        current_stroke = []
        last_stroke_index = 0
        for point in sa_vector:
            x, y, stroke_index = point
            if stroke_index != last_stroke_index:
                # Draw the previous stroke
                if len(current_stroke) >= 2:
                    draw.line(current_stroke, fill=0, width=2)
                current_stroke = []
                last_stroke_index = stroke_index
            current_stroke.append((x, y))

        if len(current_stroke) >= 2:
            draw.line(current_stroke, fill=0, width=2)

        # Create class folder
        class_folder = os.path.join(output_dir, label)
        os.makedirs(class_folder, exist_ok=True)

        # Save image
        image_path = os.path.join(class_folder, f"{img_id}.png")
        img.save(image_path)

    print(f"Images saved in '{output_dir}/<class_name>/' folders.")


if __name__ == "__main__":
    
    vectorized_data, stroke_vectors = vectorize(directory)
    print(f"Vectorized {len(vectorized_data)} items.")
    
    # Save the vectorized data to JSON files
    with open("Classical/processed_data/vectorized_data.json", 'w') as f:
        json.dump(vectorized_data, f, indent=4)
    print("Data saved to vectorized_data.json.")
    
    # Save the stroke vectors to a separate JSON file
    with open("Classical/processed_data/stroke_vectors.json", 'w') as f:
        json.dump(stroke_vectors, f, indent=4)
    print("Stroke vectors saved to stroke_vectors.json.")
    
    # Draw strokes as images
    draw_strokes_as_images(stroke_vectors, image_size=400, output_dir="Classical/Image_data")
    print("Stroke images created and saved.")
    