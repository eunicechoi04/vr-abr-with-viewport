import cv2
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO1_PATH = '/Users/eunicechoi04/Downloads/videoabr/output/sample/11_HogRider.mp4'
VIDEO2_PATH = '/Users/eunicechoi04/Downloads/videoabr/output/sample/game_saliency.mp4'
OUTPUT_PATH = 'saliency_output.mp4'

# Display settings
LABEL1 = "Original Video"
LABEL2 = "Saliency"
LABEL_BG_COLOR = (0, 0, 0)  # Black background for labels
LABEL_TEXT_COLOR = (255, 255, 255)  # White text
SEPARATOR_COLOR = (255, 255, 255)  # White vertical line
SEPARATOR_WIDTH = 4

# ==========================================
# MAIN FUNCTION
# ==========================================

def compare_videos_side_by_side(video1_path, video2_path, output_path, 
                                label1="Video 1", label2="Video 2"):
    """
    Play two videos side by side, synchronized by frame timestamp.
    
    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        output_path: Path to save comparison video
        label1: Label for first video
        label2: Label for second video
    """
    
    # Open both videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened():
        print(f"ERROR: Could not open {video1_path}")
        return
    
    if not cap2.isOpened():
        print(f"ERROR: Could not open {video2_path}")
        cap1.release()
        return
    
    # Get video properties
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("=" * 60)
    print("SIDE-BY-SIDE VIDEO COMPARISON")
    print("=" * 60)
    print(f"\nVideo 1 ({label1}):")
    print(f"  Resolution: {width1}x{height1}")
    print(f"  FPS: {fps1}")
    print(f"  Frames: {frames1}")
    print(f"  Duration: {frames1/fps1:.2f}s")
    
    print(f"\nVideo 2 ({label2}):")
    print(f"  Resolution: {width2}x{height2}")
    print(f"  FPS: {fps2}")
    print(f"  Frames: {frames2}")
    print(f"  Duration: {frames2/fps2:.2f}s")
    
    # Use the lower FPS for output
    output_fps = min(fps1, fps2)
    
    # Resize videos to same height (use minimum height)
    target_height = min(height1, height2)
    
    # Calculate aspect ratios to maintain proportions
    aspect1 = width1 / height1
    aspect2 = width2 / height2
    
    target_width1 = int(target_height * aspect1)
    target_width2 = int(target_height * aspect2)
    
    # Add label bar height
    label_height = 60
    
    # Calculate output dimensions
    output_width = target_width1 + SEPARATOR_WIDTH + target_width2
    output_height = target_height + label_height
    
    print(f"\nOutput:")
    print(f"  Resolution: {output_width}x{output_height}")
    print(f"  FPS: {output_fps}")
    print(f"  Processing: {min(frames1, frames2)} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
    
    print("\n" + "=" * 60)
    print("Processing frames...")
    print("=" * 60)
    
    frame_count = 0
    max_frames = min(frames1, frames2)
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Resize frames to target dimensions
        if (height1 != target_height) or (width1 != target_width1):
            frame1 = cv2.resize(frame1, (target_width1, target_height))
        
        if (height2 != target_height) or (width2 != target_width2):
            frame2 = cv2.resize(frame2, (target_width2, target_height))
        
        # Create label bar
        label_bar = np.zeros((label_height, output_width, 3), dtype=np.uint8)
        label_bar[:] = LABEL_BG_COLOR
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Label 1 (centered on left half)
        text_size1 = cv2.getTextSize(label1, font, font_scale, thickness)[0]
        text_x1 = (target_width1 - text_size1[0]) // 2
        text_y1 = (label_height + text_size1[1]) // 2
        cv2.putText(label_bar, label1, (text_x1, text_y1), 
                   font, font_scale, LABEL_TEXT_COLOR, thickness, cv2.LINE_AA)
        
        # Label 2 (centered on right half)
        text_size2 = cv2.getTextSize(label2, font, font_scale, thickness)[0]
        text_x2 = target_width1 + SEPARATOR_WIDTH + (target_width2 - text_size2[0]) // 2
        text_y2 = (label_height + text_size2[1]) // 2
        cv2.putText(label_bar, label2, (text_x2, text_y2), 
                   font, font_scale, LABEL_TEXT_COLOR, thickness, cv2.LINE_AA)
        
        # Add timestamp
        timestamp = f"Time: {frame_count / output_fps:.2f}s | Frame: {frame_count}"
        timestamp_size = cv2.getTextSize(timestamp, font, 0.6, 1)[0]
        timestamp_x = (output_width - timestamp_size[0]) // 2
        # cv2.putText(label_bar, timestamp, (timestamp_x, label_height - 10), 
        #            font, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Create separator line
        separator = np.ones((target_height, SEPARATOR_WIDTH, 3), dtype=np.uint8)
        separator[:] = SEPARATOR_COLOR
        
        # Combine frames horizontally: frame1 | separator | frame2
        combined_frame = np.hstack([frame1, separator, frame2])
        
        # Stack label bar on top
        final_frame = np.vstack([label_bar, combined_frame])
        
        # Write to output
        out.write(final_frame)
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{max_frames} frames ({100*frame_count/max_frames:.1f}%)")
    
    # Cleanup
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"✅ Done! Processed {frame_count} frames")
    print(f"📹 Output saved to: {output_path}")
    print(f"   Duration: {frame_count / output_fps:.2f} seconds")
    print("=" * 60)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    compare_videos_side_by_side(
        video1_path=VIDEO1_PATH,
        video2_path=VIDEO2_PATH,
        output_path=OUTPUT_PATH,
        label1=LABEL1,
        label2=LABEL2
    )
