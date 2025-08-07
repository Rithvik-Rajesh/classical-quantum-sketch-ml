import json 
import math
from scipy.spatial import ConvexHull
import pymongo
import logging
from datetime import datetime
import sys
import traceback

# Configure logging
def setup_logging():
    """Setup logging configuration with both file and console handlers."""
    log_filename = f"feature_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler]
    )
    
    return logging.getLogger(__name__)

def connect_to_mongodb(uri, db_name):
    client = pymongo.MongoClient(uri)
    db = client[db_name]
    return db

def load_data(file_path):
    """Load data from JSON file with error handling."""
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r') as file:
            data = json.load(file)
        logger.info(f"Successfully loaded {len(data)} records")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise

def validate_sa_vector(sa_vector):
    """Validate SA vector format and content."""
    if not sa_vector:
        return False, "SA vector is empty"
    
    if not isinstance(sa_vector, list):
        return False, "SA vector is not a list"
    
    for i, point in enumerate(sa_vector):
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            return False, f"Point {i} is not a valid (x, y, stroke_id) tuple"
        
        try:
            x, y, s_id = point
            float(x), float(y), int(s_id)
        except (ValueError, TypeError):
            return False, f"Point {i} contains invalid data types"
    
    return True, "Valid"

def average_stroke_length(sa_vector):
    from collections import defaultdict
    strokes = defaultdict(list)
    for x, y, s_id in sa_vector:
        strokes[s_id].append((x, y))

    total_length = 0
    for points in strokes.values():
        if len(points) < 2:
            continue
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            total_length += math.hypot(x2 - x1, y2 - y1)

    return round(total_length / len(strokes),3) if strokes else 0

def bounding_box_features(sa_vector):
    xs = [x for x, y, _ in sa_vector]
    ys = [y for x, y, _ in sa_vector]
    if not xs or not ys:
        return 0, 0, 0
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    aspect_ratio = height / width if width != 0 else 0
    
    return width, height, aspect_ratio

def centroid(sa_vector):
    xs = [x for x, _, _ in sa_vector]
    ys = [y for _, y, _ in sa_vector]
    if not xs or not ys:
        return 0, 0
    return round(sum(xs) / len(xs),3), round(sum(ys) / len(ys),3)

def compactness(sa_vector):
    if len(sa_vector) < 2:
        return 0
    perimeter = 0
    for i in range(1, len(sa_vector)):
        x1, y1, _ = sa_vector[i - 1]
        x2, y2, _ = sa_vector[i]
        perimeter += math.hypot(x2 - x1, y2 - y1)
    width, height, aspect_ratio = bounding_box_features(sa_vector)
    area = width * height
    
    return round((perimeter ** 2) / area,3) if area != 0 else 0, width, height, round(aspect_ratio,3), area, round(perimeter,3) 


def convex_hull_area(sa_vector):
    if len(sa_vector) < 3:
        return 0
    points = [(x, y) for x, y, _ in sa_vector]
    try:
        hull = ConvexHull(points)
        return round(hull.area,3)
    except:
        return 0

def symmetry(sa_vector):
    cx, cy = centroid(sa_vector)
    hor_diff = 0
    ver_diff = 0
    for x, y, _ in sa_vector:
        hor_diff += abs(y - (2 * cy - y))
        ver_diff += abs(x - (2 * cx - x))
    return round(hor_diff / len(sa_vector),3), round(ver_diff / len(sa_vector),3)

def straightness(sa_vector):
    if len(sa_vector) < 2:
        return 0
    x1, y1, _ = sa_vector[0]
    x2, y2, _ = sa_vector[-1]
    direct_dist = math.hypot(x2 - x1, y2 - y1)

    path_length = 0
    for i in range(1, len(sa_vector)):
        x1, y1, _ = sa_vector[i - 1]
        x2, y2, _ = sa_vector[i]
        path_length += math.hypot(x2 - x1, y2 - y1)
    return round(direct_dist / path_length if path_length != 0 else 0,3)

def start_end_positions(sa_vector):
    if not sa_vector:
        return 0, 0, 0, 0
    x_start, y_start, _ = sa_vector[0]
    x_end, y_end, _ = sa_vector[-1]
    return x_start, y_start, x_end, y_end

def extract_features_from_drawing(drawing):
    """Extract all features from a drawing with comprehensive error handling."""
    logger = logging.getLogger(__name__)
    
    try:
        # Validate drawing structure
        if not isinstance(drawing, dict):
            raise ValueError("Drawing is not a dictionary")
        
        required_fields = ["id", "label", "sa_vector"]
        for field in required_fields:
            if field not in drawing:
                raise KeyError(f"Missing required field: {field}")
        
        logger.debug(f"Processing drawing ID: {drawing['id']}, Label: {drawing['label']}")
        
        sa_vector = drawing["sa_vector"]
        
        # Extract all features
        compactness_value, width, height, aspect_ratio, area, perimeter = compactness(sa_vector)
        x_centroid, y_centroid = centroid(sa_vector)
        x_start, y_start, x_end, y_end = start_end_positions(sa_vector)
        h_symmetry, v_symmetry = symmetry(sa_vector)

        features = {
            "id": drawing["id"],
            "label": drawing["label"],
            "no_strokes": drawing.get("no_strokes", 0),
            "no_points": drawing.get("no_points", len(sa_vector)),
            "avg_stroke_length": average_stroke_length(sa_vector),
            "bbox_width": width,
            "bbox_height": height,
            "aspect_ratio": aspect_ratio,
            "centroid_x": x_centroid,
            "centroid_y": y_centroid,
            "compactness": compactness_value,
            "convex_hull_area": convex_hull_area(sa_vector),
            "horizontal_symmetry": h_symmetry,
            "vertical_symmetry": v_symmetry,
            "straightness": straightness(sa_vector),
            "start_x": x_start,
            "start_y": y_start,
            "end_x": x_end,
            "end_y": y_end,
            "area": area,
            "perimeter": perimeter
        }
        
        logger.debug(f"Successfully extracted features for drawing {drawing['id']}")
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features from drawing {drawing.get('id', 'unknown')}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    logger.info("Starting feature extraction process")
    
    try:
        # Load data
        file_path = "Classical/processed_data/stroke_vectors.json"
        data = load_data(file_path=file_path)
        logger.info(f"Loaded {len(data)} drawings from {file_path}")
        
        # Connect to MongoDB
        DATABASE_URI = "mongodb://localhost:27017/"
        DATABASE_NAME = "Doodle_Classifier"
        db = connect_to_mongodb(DATABASE_URI, DATABASE_NAME)
        collection = db["Extracted_Features"]
        
        # Process drawings
        successful_insertions = 0
        failed_insertions = 0
        
        for i, drawing in enumerate(data, 1):
            try:
                logger.info(f"Processing drawing {i}/{len(data)} (ID: {drawing.get('id', 'unknown')})")
                
                features = extract_features_from_drawing(drawing)
                if features is None:
                    failed_insertions += 1
                    logger.warning(f"Skipping drawing {drawing.get('id', 'unknown')} due to feature extraction failure")
                    continue
                
                # Insert into MongoDB
                result = collection.insert_one(features)
                if result.inserted_id:
                    successful_insertions += 1
                    logger.debug(f"Successfully inserted features for drawing {features['id']}")
                else:
                    failed_insertions += 1
                    logger.error(f"Failed to insert features for drawing {features['id']}")
                
            except Exception as e:
                failed_insertions += 1
                logger.error(f"Error processing drawing {i}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        # Final summary
        logger.info("=" * 50)
        logger.info("FEATURE EXTRACTION COMPLETED")
        logger.info(f"Total drawings processed: {len(data)}")
        logger.info(f"Successful insertions: {successful_insertions}")
        logger.info(f"Failed insertions: {failed_insertions}")
        logger.info(f"Success rate: {(successful_insertions/len(data)*100):.2f}%")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


