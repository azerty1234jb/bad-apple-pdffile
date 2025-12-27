import cv2
import numpy as np
import os
import glob
import json
import hashlib
#import threading
from pdf2image import convert_from_path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# ================= config =================
PDF_FOLDER = "./input_pdfs"
CACHE_FOLDER = "./cache_thumbnails"
DB_FILE = "./mosaic_db.json"
INPUT_VIDEO = "bad_apple.mp4"
OUTPUT_VIDEO = "redacted_apple.mp4"

CACHE_W, CACHE_H = 300,420

# shape match features
FEATURE_GRID_W, FEATURE_GRID_H = 12, 16 

# number of pages horizontally
GRID_WIDTH = 24

# size of each tile of pdf file
TILE_DISPLAY_W = 120
TILE_DISPLAY_H = 168
# =================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_file_hash(filepath):
    return hashlib.md5(os.path.basename(filepath).encode('utf-8')).hexdigest()

def ingest_pdfs():
    """
    scans pdf and we save the thumbnail, while experimenting we do not want to
    calculate that again and again. We also calculate the feature vector of every
    page.
    """
    ensure_dir(CACHE_FOLDER)
    
    if os.path.exists(DB_FILE):
        print("loading db!!!")
        with open(DB_FILE, 'r') as f:
            db = json.load(f)
    else:
        db = {}

    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    files_to_process = []
    
    for pdf_path in pdf_files:
        file_id = get_file_hash(pdf_path)
        if file_id not in db:
            files_to_process.append((pdf_path, file_id))

    if not files_to_process:
        print("all pdf cached")
        return db

    print(f"processing : new pdf count : {len(files_to_process)}")

    # this part written by chatgpt i have no clue why it works :pray:

    for pdf_path, file_id in tqdm(files_to_process, desc="Ingesting"):
        try:
            pages = convert_from_path(pdf_path, dpi=50, grayscale=True)
            page_data = []
            
            for i, page in enumerate(pages):
                img_filename = f"{file_id}_p{i}.jpg"
                save_path = os.path.join(CACHE_FOLDER, img_filename)
                
                # Convert to numpy array and grayscale
                img = np.array(page)
                
                thumb = cv2.resize(img, (CACHE_W, CACHE_H))
                cv2.imwrite(save_path, thumb)
                
                # calc feature vector
                feature_img = cv2.resize(img, (FEATURE_GRID_W, FEATURE_GRID_H))
                # Normalize pixel values 0-1 for easier math work on cpu
                feature_vec = feature_img.flatten() / 255.0
                
                page_data.append({
                    "img": img_filename,
                    "vec": feature_vec.tolist()
                })
            
            db[file_id] = page_data
            
            with open(DB_FILE, 'w') as f:
                json.dump(db, f)
                
        except Exception as e:
            print(f"Skipping {pdf_path}: {e}")

    return db

def prepare_knn(db):
    print("building shape index kd_tree!!!")
    
    all_vectors = []
    all_image_paths = []
    
    for file_id in db:
        for page in db[file_id]:
            #print("HERE!!!!")
            all_vectors.append(page['vec'])
            all_image_paths.append(page['img'])
            
    if not all_vectors:
        raise ValueError("No pages found in DB!")

    X = np.array(all_vectors)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
    
    return nbrs, all_image_paths

def load_images_to_ram(image_paths):
    print("load image into RAM!!!")
    images = []
    for path in tqdm(image_paths):
        full_path = os.path.join(CACHE_FOLDER, path)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (TILE_DISPLAY_W, TILE_DISPLAY_H))
        images.append(img)
    return images

def render_mosaic(nbrs, library_images):
    print("--- STARTING RENDER ---")
    
    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    vid_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect = vid_h / vid_w
    
    GRID_HEIGHT = int(GRID_WIDTH * aspect)
    
    OUT_W = GRID_WIDTH * TILE_DISPLAY_W
    OUT_H = GRID_HEIGHT * TILE_DISPLAY_H
    
    print(f"Mosaic Grid: {GRID_WIDTH} x {GRID_HEIGHT} tiles")
    print(f"Output Res:  {OUT_W} x {OUT_H}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (OUT_W, OUT_H), False)
    
    for _ in tqdm(range(total_frames), desc="Rendering"):
        ret, frame = cap.read()
        if not ret: break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        processing_w = GRID_WIDTH * FEATURE_GRID_W
        processing_h = GRID_HEIGHT * FEATURE_GRID_H
        
        resized_frame = cv2.resize(gray_frame, (processing_w, processing_h))
        
        query_vectors = []
        
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                y_start = y * FEATURE_GRID_H
                y_end = y_start + FEATURE_GRID_H
                x_start = x * FEATURE_GRID_W
                x_end = x_start + FEATURE_GRID_W
                
                tile = resized_frame[y_start:y_end, x_start:x_end]
                
                vec = tile.flatten() / 255.0
                query_vectors.append(vec)
        
        # QUERY THE KD-TREE
        distances, indices = nbrs.kneighbors(query_vectors)
        
        canvas = np.zeros((OUT_H, OUT_W), dtype=np.uint8)
        
        tile_idx = 0
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                best_match_idx = indices[tile_idx][0]
                
                pdf_tile = library_images[best_match_idx]
                
                y_pos = y * TILE_DISPLAY_H
                x_pos = x * TILE_DISPLAY_W
                canvas[y_pos:y_pos+TILE_DISPLAY_H, x_pos:x_pos+TILE_DISPLAY_W] = pdf_tile
                
                tile_idx += 1
                
        out.write(canvas)

    cap.release()
    out.release()
    print("finish")

if __name__ == "__main__":
    db = ingest_pdfs()
    knn_model, image_paths = prepare_knn(db)
    img_library = load_images_to_ram(image_paths)
    render_mosaic(knn_model, img_library)
