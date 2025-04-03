import argparse
from pathlib import Path
from PIL import Image
import sys
import os
import multiprocessing
from tqdm import tqdm

# Helper function to parse resolution string
def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parses a 'WIDTHxHEIGHT' string into a tuple of integers."""
    try:
        width, height = map(int, res_str.lower().split('x'))
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers.")
        return width, height
    except ValueError as e:
        print(f"Error: Invalid resolution format '{res_str}'. Use 'WIDTHxHEIGHT' (e.g., '1024x512'). {e}", file=sys.stderr)
        sys.exit(1)

def combine_image_pair(img_a_path: Path, img_b_path: Path, output_path: Path, output_size: tuple[int, int]):
    """Combines two images side-by-side into a single output image of the specified size."""
    try:
        img_a = Image.open(img_a_path)
        img_b = Image.open(img_b_path)

        width, height = img_a.size
        if img_a.size != img_b.size or width != height:
            print(f"Warning: Input images {img_a_path.name} and {img_b_path.name} are not identical squares ({img_a.size} vs {img_b.size}). Skipping.", file=sys.stderr)
            return False # Indicate failure/skip

        target_width, target_height = output_size
        # Calculate the size each input image should be resized to (half the width, full height)
        resize_width = target_width // 2
        resize_height = target_height

        # Resize images using a high-quality downsampling filter if needed
        try:
            img_a_resized = img_a.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
            img_b_resized = img_b.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
        except Exception as resize_e:
             print(f"Error resizing images {img_a_path.name}/{img_b_path.name} to {resize_width}x{resize_height}: {resize_e}", file=sys.stderr)
             return False


        # Create the new image canvas with the target output size
        combined_img = Image.new('RGB', (target_width, target_height)) # Use RGB, change if alpha needed

        # Paste resized image A on the left
        combined_img.paste(img_a_resized, (0, 0))

        # Paste resized image B on the right
        combined_img.paste(img_b_resized, (resize_width, 0)) # x-offset is the width of the resized image A

        combined_img.save(output_path)
        return True 

    except FileNotFoundError as e:
        print(f"Error: Image file not found: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error processing images {img_a_path.name}/{img_b_path.name}: {e}", file=sys.stderr)
        return False

def process_directories(dir_a_path: Path, dir_b_path: Path, output_dir_path: Path, output_size: tuple[int, int]):
    """
    Iterates through images in dir_a, finds corresponding images in dir_b,
    and saves combined images to output_dir using multiprocessing.
    """
    if not dir_a_path.is_dir():
        print(f"Error: Input directory A not found: {dir_a_path}", file=sys.stderr)
        sys.exit(1)
    if not dir_b_path.is_dir():
        print(f"Error: Input directory B not found: {dir_b_path}", file=sys.stderr)
        sys.exit(1)

    # Create the output directory if it doesn't exist
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir_path}")
    except OSError as e:
        print(f"Error: Could not create output directory {output_dir_path}: {e}", file=sys.stderr)
        sys.exit(1)

    tasks = []
    skipped_count_initial = 0
    print(f"Scanning directories and preparing tasks...")

    # Iterate through files in the first directory (train_A)
    for img_a_file in dir_a_path.iterdir():
        if img_a_file.is_file():
            img_b_file = dir_b_path / img_a_file.name
            if img_b_file.is_file():
                output_file = output_dir_path / img_a_file.name
                # Add arguments tuple for starmap
                tasks.append((img_a_file, img_b_file, output_file, output_size))
            else:
                print(f"Warning: Corresponding file {img_a_file.name} not found in {dir_b_path}. Skipping.", file=sys.stderr)
                skipped_count_initial += 1

    if not tasks:
        print("No image pairs found to process.")
        if skipped_count_initial > 0:
             print(f"Skipped {skipped_count_initial} files due to missing pairs.")
        return # Exit if no tasks

    print(f"Found {len(tasks)} image pairs to process. Starting parallel processing...")

    processed_count = 0
    skipped_count_processing = 0

    # Determine the number of processes to use (default to CPU count)
    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} worker processes.")


    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.starmap(combine_image_pair, tasks), total=len(tasks), desc="Combining Images"))

    # Process results
    for result in results:
        if result:
            processed_count += 1
        else:
            skipped_count_processing += 1

    total_skipped = skipped_count_initial + skipped_count_processing

    print(f"\nProcessing complete.")
    print(f"Successfully combined {processed_count} image pairs.")
    if total_skipped > 0:
        print(f"Skipped {total_skipped} pairs (initial scan + processing errors/skips).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine pairs of images from two input directories side-by-side."
    )
    parser.add_argument(
        "--dir_a",
        type=str,
        required=True,
        help="Path to the first input directory."
    )
    parser.add_argument(
        "--dir_b",
        type=str,
        required=True,
        help="Path to the second input directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for combined images."
    )
    parser.add_argument(
        "--output_resolution",
        type=str,
        default="1024x512",
        help="Resolution of the output combined image in WIDTHxHEIGHT format (default: 1024x512)."
    )

    args = parser.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)
    output_dir = Path(args.output_dir)
    # Parse the resolution argument
    try:
        output_size = parse_resolution(args.output_resolution)
    except SystemExit: # Catch exit from parse_resolution on error
        sys.exit(1)

    process_directories(dir_a, dir_b, output_dir, output_size)