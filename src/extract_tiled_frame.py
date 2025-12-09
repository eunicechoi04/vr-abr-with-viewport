import cv2
import numpy as np
import argparse

# ==========================================
# CONFIGURATION
# ==========================================
ROWS = 4
COLS = 6
TOTAL_TILES = 24

# Visualization Settings
HIGH_QUALITY_COLOR = (0, 255, 0)  # Green box for tiles
BORDER_THICKNESS = 1

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_tile_rect(tile_index, img_w, img_h, rows, cols):
    """
    Returns (x, y, w, h) for a specific tile index.
    Assumes row-major ordering (0 is top-left, rows-1 is bottom-right).
    """
    tile_w = img_w // cols
    tile_h = img_h // rows

    # Calculate row and col for this index
    r = tile_index // cols
    c = tile_index % cols

    x = c * tile_w
    y = r * tile_h

    return x, y, tile_w, tile_h

def tile_image(image_path, output_path, tiles_to_highlight=None):
    """
    Tile an image and highlight specified tiles in green.

    Args:
        image_path: Path to the input image file
        output_path: Path to save the output image
        tiles_to_highlight: List of tile indices to highlight (0-23). If None, highlights all tiles.
    """
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image file: {image_path}")
        return False

    height, width = frame.shape[:2]
    
    print(f"Image info:")
    print(f"  Resolution: {width}x{height}")
    
    return _process_and_save_tiled_frame(frame, output_path, tiles_to_highlight, frame_info=f"Image: {image_path}")


def extract_and_tile_frame(video_path, frame_number, output_path, tiles_to_highlight=None):
    """
    Extract frame n from video, tile it, and highlight specified tiles in green.

    Args:
        video_path: Path to the input video file
        frame_number: Frame number to extract (0-indexed)
        output_path: Path to save the output image
        tiles_to_highlight: List of tile indices to highlight (0-23). If None, highlights all tiles.
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")

    # Validate frame number
    if frame_number < 0 or frame_number >= total_frames:
        print(f"Error: Frame number {frame_number} is out of range (0-{total_frames-1})")
        cap.release()
        return False

    # Seek to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return False

    print(f"\nExtracted frame {frame_number}")
    
    return _process_and_save_tiled_frame(frame, output_path, tiles_to_highlight, frame_info=f"Frame: {frame_number}")


def _process_and_save_tiled_frame(frame, output_path, tiles_to_highlight, frame_info):
    """
    Internal function to process frame/image with tile overlay.
    
    Args:
        frame: Input image/frame
        output_path: Path to save output
        tiles_to_highlight: Tiles to highlight
        frame_info: Info text to display
    """
    height, width = frame.shape[:2]

    # If no tiles specified, highlight all tiles
    if tiles_to_highlight is None:
        tiles_to_highlight = list(range(TOTAL_TILES))

    # Create output frame
    output_frame = frame.copy()

    # Draw tile grid with green borders for specified tiles
    for tile_idx in range(TOTAL_TILES):
        tx, ty, tw, th = get_tile_rect(tile_idx, width, height, ROWS, COLS)

        if tile_idx in tiles_to_highlight:
            # Draw green border for highlighted tiles
            cv2.rectangle(output_frame, (tx, ty), (tx+tw, ty+th),
                        HIGH_QUALITY_COLOR, BORDER_THICKNESS)
        else:
            # Draw subtle gray border for non-highlighted tiles
            cv2.rectangle(output_frame, (tx, ty), (tx+tw, ty+th),
                        (128, 128, 128), 2)

    # Add info overlay
    # info_text = f"{frame_info} | Tiles: {ROWS}x{COLS} | Highlighted: {len(tiles_to_highlight)}/{TOTAL_TILES}"
    # cv2.putText(output_frame, info_text, (20, 40),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Save the output image
    cv2.imwrite(output_path, output_frame)
    print(f"\nSaved tiled image to: {output_path}")
    print(f"Highlighted {len(tiles_to_highlight)} tiles in green")

    return True

# ==========================================
# MAIN
# ==========================================

def main():
    # Example 1: Extract frame from video
    video_path = '/Users/eunicechoi04/Downloads/videoabr/data/videos/2bpICIClAIg.webm'
    frame_number = 1835
    output_path = '/Users/eunicechoi04/Downloads/videoabr/output/tiled_frame.png'
    image_path = '/Users/eunicechoi04/Downloads/videoabr/src/25555B72-7DEC-4B43-A6E1-3296BFCC1C64.png'
    # extract_and_tile_frame(
    #     video_path=video_path,
    #     frame_number=frame_number,
    #     output_path=output_path,
    # )
    
    # Example 2: Tile an existing image (uncomment to use)
    # image_path = '/Users/eunicechoi04/Downloads/videoabr/output/sample_image.png'
    # output_path = '/Users/eunicechoi04/Downloads/videoabr/output/tiled_image.png'
    # tiles_to_highlight = [14, 15, 8, 9, 13, 20, 21, 7, 16]  # Top 9 tiles
    # 
    tile_image(
        image_path=image_path,
        output_path=output_path,
    )

if __name__ == "__main__":
    main()
