import cv2
import numpy as np
import os

def create_sparse_motion_map(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mkv', '.webm'))]

    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)
        output_filename = f"sparse_motion_{os.path.splitext(video_file)[0]}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"Processing: {video_file}...")

        cap = cv2.VideoCapture(input_path)
        
        # Methodology constraint: 30 FPS for 60 seconds
        TARGET_FPS = 30
        MAX_FRAMES = 60 * TARGET_FPS 
        
        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (width, height), isColor=False)

        # Read first frame
        ret, old_frame = cap.read()
        if not ret:
            cap.release()
            continue
            
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # PARAMETERS FOR FEATURE TRACKING (The "Sparse" part)
        # This decides what counts as a "feature" (corner/edge)
        feature_params = dict(maxCorners=20000,   # Track up to 20,000 points
                              qualityLevel=0.01,  # Lower = more points
                              minDistance=3,      # Min distance between points
                              blockSize=7)

        # PARAMETERS FOR LUCAS-KANADE OPTICAL FLOW
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        frame_count = 0

        while frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Find features in the PREVIOUS frame to track
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

            if p0 is not None:
                # 2. Calculate Optical Flow (track p0 to p1)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                # 3. Select good points (status == 1 means successfully tracked)
                if p1 is not None:
                    good_new = p1[st == 1]
                    
                    # 4. Create an empty black image
                    motion_map = np.zeros_like(old_gray)
                    
                    # 5. Draw white dots at the NEW positions of moving features
                    # The paper says "white pixel indicates the pixel is on one of the optical flows"
                    for i, (new) in enumerate(good_new):
                        a, b = new.ravel()
                        # Use standard drawing to mark the pixel
                        # circle with radius 1 or 2 simulates the "thickness" seen in your images
                        cv2.circle(motion_map, (int(a), int(b)), 1, 255, -1)
                    
                    out.write(motion_map)
                else:
                    # If no flow found, write black frame
                    out.write(np.zeros_like(old_gray))
            else:
                out.write(np.zeros_like(old_gray))

            # Update for next iteration
            old_gray = frame_gray.copy()
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  > Processed {frame_count} frames...")

        cap.release()
        out.release()
        print(f"Finished {video_file}")

if __name__ == "__main__":
    INPUT_DIR = "/Users/eunicechoi04/Downloads/videoabr/data/videos"
    OUTPUT_DIR = "/Users/eunicechoi04/Downloads/videoabr/output/motion_maps"
    create_sparse_motion_map(INPUT_DIR, OUTPUT_DIR)