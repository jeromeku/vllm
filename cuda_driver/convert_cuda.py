# Open a specified file and convert every byte to a list of unsigned chars in hexadecimal format,
# then write the list to a new text file.
# Usage: python convert_file_to_chars.py --input_bin_file <input_file> --output_char_file <output_file>

import argparse
import os
from pathlib import Path
DEFAULT_OUTPUT_DIR = "binaries"
def main():

    parser = argparse.ArgumentParser(
        description='Convert a binary file to a list of unsigned chars')
    parser.add_argument('--input', '-i',
                        type=Path,
                        help='The input binary file')
    parser.add_argument('--output', '-o',
                        type=str,
                        default=None,
                        help='The output char file')

    args = parser.parse_args()
    if args.output is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        output_path = Path(DEFAULT_OUTPUT_DIR) / Path(args.input.stem)
        args.output = output_path.with_suffix(".bin")

    if not os.path.isfile(args.input):
        print('The input binary file does not exist')
        return

    with open(args.input, 'rb') as f:
        content = f.read()
    breakpoint()
    chars = list(content)
    # Convert each char to a string in hexadecimal format "0x??"
    chars = ['0x{:02x}'.format(c) for c in chars]
    with open(args.output, 'w') as f:
        f.write(', '.join(chars))
    print(f"Wrote to {args.output}")

if __name__ == "__main__":

    main()