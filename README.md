# Segment-App

# Depth Segmentation App

## Overview

The Segmentation App is a service that generates depth maps and performs semantic segmentation on images. This project leverages the Depth-Anything-V2 model for depth map generation and the Segformer model for semantic segmentation. The application is built using Flask. Recent updates include the addition of performance metrics (IoU and Dice Coefficient) and the segmentation process has been refined with pre-processing steps.

## Features

- **Depth Map Generation**: Generate depth maps from input images using the Depth-Anything-V2 model.
- **Semantic Segmentation**: Perform semantic segmentation to identify specific objects in images using the Segformer model.
- **Overlay Results**: Overlay segmentation results on the original images for visualization.
- **Performance Metrics**: Calculate and display IoU and Dice Coefficient to evaluate segmentation accuracy.
- **Pre-processing**: Pre-process images using K-Means clustering and image augmentation for enhanced segmentation results.

## Requirements
- Docker
- Python 3.9+
- Flask
- Torch 2.4.0
- Transformers
- OpenCV

## Setup

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/sarahbanat/Segment-App.git
cd depth-segmentation-app
```


### Build the Docker Image

Build the Docker image using the provided Dockerfile:

```bash
docker build -t depth-segmentation-app .
```

### Run the Docker Container
Run the Docker container, mapping the port and mounting a local directory for outputs:

```bash
docker run -p 5001:5001 -v /path/to/local/output/dir:/tmp depth-segmentation-app
```
_Replace /path/to/local/output/dir with the path to a local directory where you want the output files to be saved._

## Usage
### Upload and Segment an Image
1.	Ensure the Docker container is running.
2.	Open a terminal and run the following command to test the segmentation:
 
```bash
curl -X POST -F "file=@/path/to/your/image.jpg" -F "ground_truth=@/path/to/your/ground_truth.png" http://127.0.0.1:5001/segment
```

_Replace /path/to/your/image.jpg with the path to the image you want to process. And /path/to/your/ground_truth.png  with the ground truth location_

### Accessing Results
The segmentation results will be saved in the directory you mounted to /tmp in the Docker container. You can find the following files:

•	Segmentation Result: /path/to/local/output/dir/output_seg.png

•	Overlay Result: /path/to/local/output/dir/output_overlay.png
 
## Project Structure
```
Segment-App/
│
├── Depth-Anything-V2/         # Dir that has the Depth-Anything-V2 model and related files
│   ├── checkpoints/
│   ├── assets/
│   ├── metric_depth/
│   ├── DA-2K.md
│   ├── LICENSE
│   ├── README.md
│   ├── app.py
│   ├── run.py
│   └── run_video.py
│
├── data/                      # Dir to store input images and output results
│   ├── depthmaps/
│   ├── depthmaps_gs/
│   ├── imgs/
│   └── test/
│       ├── output/
│       ├── test1.jpg
│       ├── output_overlay.png
│       └── output_seg.png
│
├── app/
│   ├── __init__.py            # Flask initialization file
│   ├── depth_map.py           # Depth map generation logic
│   ├── evaluation.py          # Evaluation metrics (IoU and Dice Coefficient)
│   ├── main.py                # Main entry point for the Flask app
│   ├── routes.py              # API routes
│   ├── segmentation.py        # Segmentation and pre-processing logic
│   └── utils.py               # Utility functions (e.g., image processing)
│
├── Dockerfile                 # Dockerfile to build the Docker image
├── entrypoint.sh              # Entrypoint script for Docker container
├── environment.yml            # Conda environment configuration
├── requirements.txt           # Python dependencies
└── README.md                  # Project README
 ```
## Development
### Running Locally
To run the application locally without Docker, follow these steps:
	1.	Create a virtual environment and activate it:
 
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

  2.	Install dependencies:
     
 ```bash
pip install -r requirements.txt
```

  3.	Run the Flask application:
     
 ```bash
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5001
```

## Acknowledgements

•	Depth-Anything-V2 - Depth map generation model.
•	Segformer - Semantic segmentation model.

