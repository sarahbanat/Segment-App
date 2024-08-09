import os
import subprocess

def generate_depth_map(image_path, output_dir):
    try:
        abs_image_path = os.path.abspath(image_path)
        abs_output_dir = os.path.abspath(output_dir)

        if not os.path.exists(abs_output_dir):
            os.makedirs(abs_output_dir)

        depth_anything_v2_dir = "/app/Depth-Anything-V2"
        cmd = [
            "python", os.path.join(depth_anything_v2_dir, "run.py"),
            "--encoder", "vitl",
            "--img-path", abs_image_path,
            "--outdir", abs_output_dir,
            "--pred-only"
        ]

        result = subprocess.run(cmd, check=True, cwd=depth_anything_v2_dir, capture_output=True, text=True, timeout=300)

        depth_map_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        depth_map_path = os.path.join(abs_output_dir, depth_map_filename)
        return depth_map_path
    except subprocess.CalledProcessError as e:
        return None
    except subprocess.TimeoutExpired as e:
        return None