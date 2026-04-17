import os
import subprocess
import json
import time



def load_converted_data(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {}

def save_converted_data(json_path, data):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

def get_file_mod_time(path):
    return os.path.getmtime(path)

def convert_all_svgs_to_pdf(input_dir, output_dir, json_path="converted_files.json", inkscape_path=r"C:\Program Files\Inkscape\bin\inkscape.exe"):
    os.makedirs(output_dir, exist_ok=True)
    converted_data = load_converted_data(json_path)
    updated_data = {}

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".svg"):
            input_svg = os.path.join(input_dir, filename)
            output_pdf = os.path.join(output_dir, os.path.splitext(filename)[0] + ".pdf")
            mod_time = get_file_mod_time(input_svg)

            # Skip if unchanged
            if filename in converted_data and converted_data[filename] == mod_time:
                print(f"Skipping (unchanged): {filename}")
                updated_data[filename] = mod_time
                continue

            # Convert
            try:
                subprocess.run([
                    inkscape_path,
                    input_svg,
                    "--export-type=pdf",
                    f"--export-filename={output_pdf}"
                ], check=True)
                print(f"Converted: {filename}")
                updated_data[filename] = mod_time
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {filename}: {e}")

    save_converted_data(json_path, updated_data)

def svg2pdf():
    # Example usage
    convert_all_svgs_to_pdf("svg", "figs")

