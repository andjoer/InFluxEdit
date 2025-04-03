import json
import argparse
from pathlib import Path
import sys

def create_prompt_files(json_file_path: Path, output_dir_path: Path, prefix: str = ""):
    """
    Reads a JSON file mapping image filenames to prompts, prepends an
    optional prefix to each prompt, and creates corresponding .txt files
    in the output directory.

    Args:
        json_file_path: Path to the input JSON file.
        output_dir_path: Path to the directory where .txt files will be saved.
        prefix: A string to prepend to each prompt. Defaults to "".
    """
    if not json_file_path.is_file():
        print(f"Error: JSON file not found at {json_file_path}", file=sys.stderr)
        sys.exit(1)

    # Create the output directory if it doesn't exist
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir_path}")
    except OSError as e:
        print(f"Error: Could not create output directory {output_dir_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Read and parse the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
        print(f"Successfully loaded JSON data from {json_file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading JSON file {json_file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(prompt_data, dict):
        print(f"Error: Expected JSON data to be a dictionary (object), but got {type(prompt_data)}", file=sys.stderr)
        sys.exit(1)

    # Iterate through the JSON data and create .txt files
    count = 0
    # Add a space to the prefix if it's not empty, for separation
    prefix_with_space = f"{prefix} " if prefix else ""

    for image_filename, prompt in prompt_data.items():
        if not isinstance(prompt, str):
            print(f"Warning: Skipping entry '{image_filename}' because its value is not a string (prompt). Found type: {type(prompt)}")
            continue

        # Get the base filename without the original extension
        base_filename = Path(image_filename).stem
        txt_filename = f"{base_filename}.txt"
        output_txt_path = output_dir_path / txt_filename

        # Combine prefix and prompt
        content_to_write = f"{prefix_with_space}{prompt}"

        try:
            # Write the prompt to the .txt file, ensuring UTF-8 encoding
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(content_to_write)
            count += 1
        except IOError as e:
            print(f"Error writing file {output_txt_path}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred while processing '{image_filename}': {e}", file=sys.stderr)


    print(f"\nSuccessfully created {count} prompt files in {output_dir_path}")
    if prefix:
        print(f"Prefixed each prompt with: '{prefix}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a JSON file of image prompts into individual .txt files, optionally adding a prefix."
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="prompts",
        help="Path to the output directory for the .txt files (default: 'prompts')."
    )
    parser.add_argument(
        "-p", "--prefix",
        type=str,
        default="",
        help="Optional string to prepend to each prompt (e.g., 'photo of')."
    )


    args = parser.parse_args()

    json_path = Path(args.json_file)
    output_path = Path(args.output_dir)
    prompt_prefix = args.prefix
    create_prompt_files(json_path, output_path, prompt_prefix)