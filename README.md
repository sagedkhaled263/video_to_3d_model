# 🎥 Video to 3D Model

A complete pipeline for converting a video into a **3D model** using **COLMAP (Structure-from-Motion)** and **Instant-NGP (NeRF)**.

The workflow extracts frames from a video, estimates camera poses using COLMAP, generates a NeRF scene using Instant-NGP, and finally exports the reconstructed scene as a 3D mesh.

> **Tested configuration:** Ubuntu 22.04 · GTX 1660 Ti 6GB · CUDA 12.6 · NVIDIA Driver 560.35

---

## 📌 Pipeline Overview

```text
Video (.mp4)
     │
     ▼
frames.py
     │
     ▼
Extracted Frames
     │
     ▼
COLMAP
 ├── Feature Extraction
 ├── Feature Matching
 ├── Sparse Reconstruction
 └── Model Conversion
     │
     ▼
COLMAP Output
     │
     ▼
colmap2nerf.py
     │
     ▼
transforms.json
     │
     ▼
Image Validation
     │
     ▼
Instant-NGP
     │
     ▼
NeRF Training
     │
     ▼
3D Mesh (.obj / .ply)
     │
     ├──► MeshLab
     │
     └──► Blender
```

---

# 🛠️ Prerequisites

| Tool / Requirement   | Purpose                                        |
| -------------------- | ---------------------------------------------- |
| NVIDIA GPU + Driver  | CUDA acceleration                              |
| CUDA Toolkit 12.x    | NVCC compiler for Instant-NGP                  |
| CMake 3.24+          | Build system                                   |
| COLMAP               | Structure-from-Motion + camera pose estimation |
| Instant-NGP          | NeRF training + mesh export                    |
| MeshLab              | View and inspect 3D meshes                     |
| Blender *(optional)* | Clean up mesh + measure volume                 |

> **Note:** The tested setup uses a GTX 1660 Ti with 6GB VRAM.

---

# 1️⃣ Install System Packages

Update Ubuntu and install the required dependencies:

```bash
sudo apt update

sudo apt install -y \
    build-essential \
    git \
    python3-dev \
    python3-pip \
    libboost-all-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    ffmpeg \
    colmap \
    meshlab \
    imagemagick
```

---

# 2️⃣ Install CUDA Toolkit

If CUDA is already installed, first check whether `nvcc` is available:

```bash
nvcc --version
```

If `nvcc` is missing or the installed version is below CUDA 12.x, install CUDA 12.6.

### Remove the old CUDA toolkit

```bash
sudo apt remove -y nvidia-cuda-toolkit
sudo apt autoremove -y
```

### Install CUDA 12.6

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb

sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt update

sudo apt install -y cuda-toolkit-12-6
```

### Add CUDA to PATH

```bash
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc

echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

source ~/.bashrc
```

### Verify CUDA

```bash
nvcc --version
```

---

# 3️⃣ Upgrade CMake

The default Ubuntu CMake version may be too old for Instant-NGP.

Upgrade CMake using pip:

```bash
pip install cmake --upgrade
```

Refresh the shell:

```bash
hash -r
```

Verify the installed version:

```bash
cmake --version
```

The required version is:

```text
CMake 3.24+
```

---

# 4️⃣ Create a Python Virtual Environment

Create a dedicated virtual environment:

```bash
python3 -m venv ~/v2model
```

Activate it:

```bash
source ~/v2model/bin/activ# video_to_3d_modelate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

Install the Python dependencies:

```bash
pip install opencv-python numpy matplotlib Pillow
```

---

# 5️⃣ Clone the Repositories

Clone the Video-to-3D-Model project:

```bash
git clone https://github.com/APEXPRE123207/Video-To-3D-Model.git
```

Clone Instant-NGP with its submodules:

```bash
git clone --recursive https://github.com/NVlabs/instant-ngp
```

> ⚠️ **Important:** Do not clone Instant-NGP into a directory whose path contains spaces.

For example, avoid:

```text
/media/user/New Volume/instant-ngp
```

If the drive contains spaces, create a symlink:

```bash
ln -s "/media/user/New Volume/instant-ngp" ~/instant-ngp
```

---

# 6️⃣ Build Instant-NGP

Navigate to the Instant-NGP directory:

```bash
cd instant-ngp
```

If your root filesystem does not have enough free space, create a temporary directory on a drive with sufficient storage:

```bash
mkdir -p /path/to/large/drive/tmp
```

### Build for GTX 1660 Ti

The GTX 1660 Ti uses CUDA compute capability **75**:

```bash
TMPDIR="/path/to/large/drive/tmp" cmake . -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_ARCHITECTURES=75
```

Build the project:

```bash
TMPDIR="/path/to/large/drive/tmp" cmake --build build \
    --config RelWithDebInfo \
    -j$(nproc)
```

### Common CUDA architectures

| GPU                         | CUDA Architecture |
| --------------------------- | ----------------: |
| GTX 1660 Ti / RTX 2060–2080 |              `75` |
| RTX 3060–3090               |              `86` |
| RTX 4060–4090               |              `89` |

### Verify the build

```bash
ls build/instant-ngp
```

The Instant-NGP executable should be present.

---

# 7️⃣ Copy and Fix Project Scripts

Navigate to the Instant-NGP directory:

```bash
cd instant-ngp
```

Copy the project scripts:

```bash
cp /path/to/Video-To-3D-Model/frames.py .

cp /path/to/Video-To-3D-Model/visualize_COLMAP.py .

cp /path/to/Video-To-3D-Model/blender_volume.py .
```

## Create `video_to_COLMAP.py`

The original script uses Windows paths, so create a Linux-compatible version.

Create:

```text
instant-ngp/video_to_COLMAP.py
```

Use:

```python
import subprocess
from frames import extract_frames
from visualize_COLMAP import plot_camera_positions_with_direction
import shutil
import os

if __name__ == "__main__":
    video_path = input(
        "Insert path to the video file, including the file itself.\n"
        "Enter here: "
    )

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    frames_save_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "v2obj",
        "images"
    )

    if not os.path.exists(frames_save_path):
        os.makedirs(frames_save_path)

    number_of_samples = extract_frames(
        video_path,
        frames_save_path
    )

    print("{} samples created".format(number_of_samples))

    images_path = frames_save_path

    colmap2nerf_script = os.path.join(
        PROJECT_ROOT,
        "scripts",
        "colmap2nerf.py"
    )

    colmap2nerf_arguments = [
        "--colmap_matcher", "exhaustive",
        "--run_colmap",
        "--aabb_scale", "32",
        "--images", images_path
    ]

    command = ["python", colmap2nerf_script] + colmap2nerf_arguments

    subprocess.run(command)

    _flag = input(
        "Do you want to visualize the camera position "
        "from COLMAP? (y/n): "
    )

    _flag = True if _flag == "y" or _flag == "Y" else False

    if _flag:
        with open("transforms.json", "r") as file:
            json_data = file.read()

        plot_camera_positions_with_direction(json_data)

    transforms_file = "transforms.json"

    destination_dir = os.path.join(
        PROJECT_ROOT,
        "data",
        "v2obj"
    )

    if os.path.exists(transforms_file):

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        shutil.move(
            transforms_file,
            destination_dir
        )
```

---

## Create `correction1.py`

Place this file inside:

```text
data/v2obj/correction1.py
```

Check the original script for Windows-style backslashes:

```text
\
```

Replace them with Linux-compatible forward slashes:

```text
/
```

---

## Create `correction2.py`

Place this file inside:

```text
data/v2obj/correction2.py
```

```python
from PIL import Image
import os

# Update this path to match your setup
image_dir = "./images"

for fname in os.listdir(image_dir):
    fpath = os.path.join(image_dir, fname)

    try:
        with Image.open(fpath) as img:
            img.verify()

        print(f"OK: {fname}")

    except Exception as e:
        print(f"ERROR: {fname} - {e}")
```

---

# 8️⃣ Record Your Video

For the best reconstruction results, follow these guidelines:

* **Resolution:** 1080p 60fps or 4K preferred
* Start from the **top of the object**
* Slowly spiral downward
* Complete a full **360° orbit**
* Cover every angle of the object
* Move slowly and smoothly
* Avoid shaking
* Avoid jump cuts
* Use even lighting
* Minimize shadows
* Keep the object centered in the frame

A good capture is critical because COLMAP needs enough visual information and overlap between frames to estimate camera positions reliably.

---

# 9️⃣ Extract Frames

Activate the virtual environment:

```bash
source ~/v2model/bin/activate
```

Navigate to Instant-NGP:

```bash
cd /path/to/instant-ngp
```

Run:

```bash
python video_to_COLMAP.py
```

When prompted:

### 1. Enter the full path to your video

Example:

```text
/home/user/videos/object.mp4
```

### 2. Enter the sample rate

Use:

```text
1
```

This extracts:

```text
1 frame per second
```

> ⚠️ **Note:** If COLMAP crashes with a Qt error while running inside the virtual environment, run COLMAP manually outside the virtual environment as described in Step 10.

---

# 🔟 Run COLMAP Manually

If COLMAP fails in Step 9 because of a Qt conflict, deactivate the Python virtual environment:

```bash
deactivate
```

Then navigate to Instant-NGP:

```bash
cd /path/to/instant-ngp
```

The issue occurs because OpenCV's Qt libraries inside the virtual environment can conflict with COLMAP's Qt libraries.

## Feature Extraction

```bash
colmap feature_extractor \
    --ImageReader.camera_model OPENCV \
    --ImageReader.single_camera 1 \
    --SiftExtraction.estimate_affine_shape=true \
    --SiftExtraction.domain_size_pooling=true \
    --database_path colmap.db \
    --image_path data/v2obj/images
```

## Feature Matching

```bash
colmap exhaustive_matcher \
    --database_path colmap.db
```

## Sparse Reconstruction

Create the output directory:

```bash
mkdir -p colmap_sparse
```

Run the mapper:

```bash
colmap mapper \
    --database_path colmap.db \
    --image_path data/v2obj/images \
    --output_path colmap_sparse
```

## Export COLMAP Model to Text

Create the output directory:

```bash
mkdir -p colmap_text
```

Convert the model:

```bash
colmap model_converter \
    --input_path colmap_sparse/0 \
    --output_path colmap_text \
    --output_type TXT
```

---

# 1️⃣1️⃣ Generate `transforms.json`

Activate the virtual environment again:

```bash
source ~/v2model/bin/activate
```

Navigate to Instant-NGP:

```bash
cd /path/to/instant-ngp
```

Run:

```bash
python scripts/colmap2nerf.py \
    --images data/v2obj/images \
    --colmap_db colmap.db \
    --text colmap_text \
    --aabb_scale 32 \
    --out data/v2obj/transforms.json
```

This converts the COLMAP reconstruction into the `transforms.json` format required by Instant-NGP.

---

# 1️⃣2️⃣ Validate Images

Navigate to the scene directory:

```bash
cd /path/to/instant-ngp/data/v2obj
```

Run:

```bash
python correction1.py
```

Then:

```bash
python correction2.py
```

All images should be successfully recognized.

Remove any images reported as corrupted or invalid.

---

# 1️⃣3️⃣ Optional — Resize Frames for Low VRAM

If you are using a **6GB GPU** and have many frames, such as 300+ frames or 4K images, resizing the images can reduce VRAM usage.

Navigate to the image directory:

```bash
cd /path/to/instant-ngp/data/v2obj/images
```

Resize images to 50%:

```bash
for f in *.jpg *.png; do
    convert "$f" -resize 50% "$f"
done
```

This is particularly useful for GPUs with limited VRAM.

---

# 1️⃣4️⃣ Launch Instant-NGP and Train the NeRF

Navigate to Instant-NGP:

```bash
cd /path/to/instant-ngp
```

Launch the scene:

```bash
./build/instant-ngp --scene data/v2obj
```

The Instant-NGP GUI should open and begin real-time NeRF training.

Allow approximately:

```text
1–5 minutes
```

for the scene to become sharp.

---

# 1️⃣5️⃣ Export the 3D Mesh

Inside the Instant-NGP GUI:

1. Adjust the **crop box** to tightly surround the object.
2. Set the mesh resolution.
3. Start with a resolution of **256**.
4. Click **Export mesh**.
5. The resulting `.obj` or `.ply` file will be saved inside:

```text
data/v2obj/
```

---

# 1️⃣6️⃣ View the Mesh in MeshLab

Open the generated mesh using MeshLab:

```bash
meshlab /path/to/instant-ngp/data/v2obj/mesh.obj
```

MeshLab can be used to inspect the reconstructed geometry.

---

# 1️⃣7️⃣ Optional — Clean Up in Blender

Install Blender:

```bash
sudo apt install -y blender
```

Launch it:

```bash
blender
```

Inside Blender:

1. Go to **File → Import → Wavefront (.obj)**
2. Import the generated `.obj` file.
3. Delete floating noise artifacts.
4. Press **N** to open the sidebar.
5. Adjust the model dimensions if required.
6. Open the **Scripting** tab.
7. Create a new script.
8. Paste `blender_volume.py`.
9. Run the script.
10. The calculated volume will be printed in the Blender console.

---

# 📁 File Reference

| File                     | Purpose                                                    |
| ------------------------ | ---------------------------------------------------------- |
| `frames.py`              | Extracts video frames at the selected FPS                  |
| `video_to_COLMAP.py`     | Orchestrates frame extraction → COLMAP → `transforms.json` |
| `visualize_COLMAP.py`    | Plots camera positions from COLMAP output                  |
| `correction1.py`         | Fixes image paths in `transforms.json`                     |
| `correction2.py`         | Validates that extracted images are not corrupted          |
| `blender_volume.py`      | Computes the volume of the 3D model in Blender             |
| `scripts/colmap2nerf.py` | Converts COLMAP output into Instant-NGP `transforms.json`  |

---

# 🔄 Complete Pipeline

The complete workflow is:

```text
Video (.mp4)
    │
    ▼
frames.py
    │
    ▼
Extracted Frames
data/v2obj/images/
    │
    ▼
COLMAP feature_extractor
    │
    ▼
colmap.db
    │
    ▼
COLMAP exhaustive_matcher
    │
    ▼
COLMAP mapper
    │
    ▼
colmap_sparse/
    │
    ▼
COLMAP model_converter
    │
    ▼
colmap_text/
    │
    ▼
colmap2nerf.py
    │
    ▼
transforms.json
    │
    ▼
correction1.py + correction2.py
    │
    ▼
Validated Scene
    │
    ▼
Instant-NGP
    │
    ▼
NeRF Training
    │
    ▼
Export Mesh
    │
    ├───────────────┐
    ▼               ▼
  MeshLab         Blender
  View/Inspect    Clean Up
                  + Volume
```

---

# 🐛 Troubleshooting

| Problem                                 | Solution                                                                     |
| --------------------------------------- | ---------------------------------------------------------------------------- |
| `No CMAKE_CUDA_COMPILER found`          | Install CUDA Toolkit: `sudo apt install cuda-toolkit-12-6`                   |
| `CUDA_ARCHITECTURES is empty`           | Add `-DCMAKE_CUDA_ARCHITECTURES=75` to the CMake command                     |
| `No space left on device`               | Set `TMPDIR` to a drive with sufficient free space                           |
| CMake symlink fails with spaces in path | Use a path without spaces or create a symlink                                |
| COLMAP Qt crash inside venv             | Run COLMAP commands outside the virtual environment                          |
| `correction2.py` Windows path error     | Change the image path to `./images`                                          |
| Black / empty NeRF scene                | Check `transforms.json` and make sure the image paths match the actual files |
| CUDA out of memory on a 6GB GPU         | Resize images to 50% and keep the number of frames below approximately 200   |

---

# 📌 Important Notes

### GPU Architecture

Make sure the CUDA architecture specified during the Instant-NGP build matches your GPU.

For a GTX 1660 Ti:

```bash
-DCMAKE_CUDA_ARCHITECTURES=75
```

### Storage

Instant-NGP compilation can require significant temporary storage. If your root partition is full, use another drive for `TMPDIR`.

### Image Quality

The quality of the final 3D reconstruction depends heavily on the input video. Smooth camera movement, complete object coverage, sufficient lighting, and overlapping views are important for successful reconstruction.

### VRAM

For a 6GB GPU, large numbers of high-resolution frames can cause CUDA out-of-memory errors. Reducing image resolution and the number of frames can help.

---

# 📚 References

* [Video-To-3D-Model](https://github.com/APEXPRE123207/Video-To-3D-Model)
* [Instant-NGP](https://github.com/NVlabs/instant-ngp)
* [COLMAP](https://colmap.github.io/)
* [MeshLab](https://www.meshlab.net/)
* [Blender](https://www.blender.org/)

---

## 🚀 Quick Start

If all prerequisites are already installed, the shortest version of the workflow is:

```bash
# Activate environment
source ~/v2model/bin/activate

# Go to Instant-NGP
cd /path/to/instant-ngp

# Extract frames + run COLMAP
python video_to_COLMAP.py

# Validate images
cd data/v2obj
python correction1.py
python correction2.py

# Launch Instant-NGP
cd ../..
./build/instant-ngp --scene data/v2obj
```

Then use the Instant-NGP GUI to train the NeRF and **export the final 3D mesh**.
