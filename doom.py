import cv2
import numpy as np
import os
import glob
import json
import threading
import time
from mss import mss
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ================= config =================
CACHE_FOLDER = "./cache_thumbnails"
DB_FILE = "./mosaic_db.json"

GRID_WIDTH = 32

TILE_DISPLAY_W = 30 
TILE_DISPLAY_H = 42

FEATURE_GRID_W, FEATURE_GRID_H = 12, 16 

CAPTURE_AREA = {"top": 0, "left": 0, "width": 900, "height": 590}

TARGET_FPS = 30
# =================================================

class ScreenGrabber:
    def __init__(self, region):
        self.region = region
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.t = threading.Thread(target=self._grab_loop)
        self.t.start()

    def _grab_loop(self):
        with mss() as sct:
            while self.running:
                try:
                    img = np.array(sct.grab(self.region))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                    
                    with self.lock:
                        self.frame = gray
                except Exception as e:
                    print(f"Capture error: {e}")
                
                time.sleep(0.005) 

    def get_latest(self):
        with self.lock:
            if self.frame is None: return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.t.join()

def load_db_and_model():
    print("loading db...")
    with open(DB_FILE, 'r') as f:
        db = json.load(f)

    all_vectors = []
    all_img_paths = []

    for file_id in db:
        for page in db[file_id]:
            all_vectors.append(page['vec'])
            all_img_paths.append(page['img'])

    print(f"training KNN on {len(all_vectors)} pages!!!")
    X = np.array(all_vectors, dtype=np.float32)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', leaf_size=30).fit(X)
    
    return nbrs, all_img_paths

def load_image_tensor(image_paths):
    print("image load into RAM")
    print(2)
    count = len(image_paths)
    image_tensor = np.zeros((count, TILE_DISPLAY_H, TILE_DISPLAY_W), dtype=np.uint8)
    
    for i, path in enumerate(tqdm(image_paths)):
        full_path = os.path.join(CACHE_FOLDER, path)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized = cv2.resize(img, (TILE_DISPLAY_W, TILE_DISPLAY_H))
            image_tensor[i] = resized
            
    return image_tensor

def main():
    nbrs, image_paths = load_db_and_model()
    image_tensor = load_image_tensor(image_paths)
    
    cap_w, cap_h = CAPTURE_AREA["width"], CAPTURE_AREA["height"]
    aspect = cap_h / cap_w
    GRID_HEIGHT = int(GRID_WIDTH * aspect)
    
    process_w = GRID_WIDTH * FEATURE_GRID_W
    process_h = GRID_HEIGHT * FEATURE_GRID_H
    
    print(f"capturing: {cap_w}x{cap_h}")
    print(f"matrix init: {GRID_WIDTH}x{GRID_HEIGHT} ({GRID_WIDTH*GRID_HEIGHT} tiles)")
    
    grabber = ScreenGrabber(CAPTURE_AREA)
    grabber.start()
    
    print("start!!!! 1") 
    try:
        while True:
            start_time = time.time()
            
            frame = grabber.get_latest()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # show exactly what we are capturing
            debug_view = cv2.resize(frame, (400, int(400 * aspect)))
            cv2.imshow("input", debug_view)

            small = cv2.resize(frame, (process_w, process_h), interpolation=cv2.INTER_LINEAR)
            tiles = small.reshape(GRID_HEIGHT, FEATURE_GRID_H, GRID_WIDTH, FEATURE_GRID_W)
            tiles = tiles.transpose(0, 2, 1, 3).reshape(-1, FEATURE_GRID_H * FEATURE_GRID_W)
            vectors = tiles / 255.0
            
            _, indices = nbrs.kneighbors(vectors)
            indices = indices.flatten()
            
            # stich image
            selected_pdfs = image_tensor[indices]
            grid = selected_pdfs.reshape(GRID_HEIGHT, GRID_WIDTH, TILE_DISPLAY_H, TILE_DISPLAY_W)
            grid = grid.transpose(0, 2, 1, 3)
            final_canvas = grid.reshape(GRID_HEIGHT * TILE_DISPLAY_H, GRID_WIDTH * TILE_DISPLAY_W)
            
            cv2.imshow("final", cv2.resize(final_canvas, (900, 580)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            elapsed = time.time() - start_time
            wait = max(0.001, (1/TARGET_FPS) - elapsed)
            time.sleep(wait)

    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
