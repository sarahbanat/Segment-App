# Segmentation-App

# Depth Segmentation App

## Overview

The Depth Segmentation App is a web-based application that generates depth maps and performs semantic segmentation on images. This project leverages the Depth-Anything-V2 model for depth map generation and the Segformer model for semantic segmentation. The application is built using Flask for the web server and integrates with PyTorch and Hugging Face's transformers.

## Features

- **Depth Map Generation**: Generate depth maps from input images using the Depth-Anything-V2 model.
- **Semantic Segmentation**: Perform semantic segmentation to identify specific objects in images using the Segformer model.
- **Overlay Results**: Overlay segmentation results on the original images for visualization.

## Requirements

- Docker

## Setup

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/USERNAME/depth-segmentation-app.git
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
curl -X POST -F "file=@/path/to/your/image.jpg" http://127.0.0.1:5001/segment
```

_Replace /path/to/your/image.jpg with the path to the image you want to process._

### Accessing Results
The segmentation results will be saved in the directory you mounted to /tmp in the Docker container. You can find the following files:

•	Segmentation Result: /path/to/local/output/dir/output_seg.png

•	Overlay Result: /path/to/local/output/dir/output_overlay.png
 
## Project Structure
```
depth-segmentation-app/
│
├── Depth-Anything-V2/         # Directory containing the Depth-Anything-V2 model and related files
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
├── data/                      # Directory to store input images and output results
│   ├── depthmaps/
│   ├── depthmaps_gs/
│   ├── imgs/
│   └── test/
│       ├── output/
│       ├── test1.jpg
│       ├── output_overlay.png
│       └── output_seg.png
│
├── src/
│   ├── app.py                 # Main Flask application file
│
├── Dockerfile                 # Dockerfile to build the Docker image
├── entrypoint.sh              # Entrypoint script for Docker container
├── environment.yml            # Conda environment configuration
└── requirements.txt           # Python dependencies
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
