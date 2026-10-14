# Flash-SewerNet
This is the official implementation of the paper **"Flash SewerNet: A Lightweight Expert Network with Gaussian-Multimodal Fusion for Sewer Defect Recognition"**.

## 🚀 Overview
This network synergizes Gaussian modeling with multimodal priors, paired with an expert-assisted multi-label recognition head. This architecture is specifically co-designed for high stability and computational efffciency, making it suitable for real-time video analysis on resource-limited edge devices.

## 📊 Datasets and Resources
The experiments in the paper are conducted on the **Sewer-ML**、**XR-Sewer** and **QV-Pipe** datasets.
### 1. Material Labels & Pre-trained Weights

**The following crucial resource files are too large to host directly on GitHub. Please download them from our Baidu NetDisk (百度网盘) repository:**

- **Link:** https://pan.baidu.com/s/1FDBrb40Q1hKOqwLg_eVHzw
- **Access Code/Extraction Code (提取码):** st6m

#### **Files to Download:**
﻿
After downloading, please place them in the corresponding directories in your local project.

1.  **Material Label Files (Place in `Resources/` directory):**
    - sewer_material_labels_part1.csv and sewer_material_labels_part2.csv: Automatically generated material labels for the Sewer-ML dataset.
    - qvpipe_material_labels.csv: Automatically generated material labels for the QV-Pipe dataset.
  
2.  **Pre-trained Model Weights (Place in `Resources/` directory):**
    - Flash SewerNet.ckpt : The final Sewer_ML pre-trained weights of our SewerMan model for inference.
    - ViT-B-32.pt : The pre-trained weights of our CLIP model.
    - - Convnet.ckpt : The pre-trained weights of our Convnet-base model.
      - 
## 🛠️ Installation & Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/zkkzxk/Flash-SewerNet.git
    cd SewerMan
    ```

2.  **Create a conda environment and install dependencies:**
    ```bash
    conda create -n sewerman python=3.9
    conda activate sewerman
    pip install pytorch_lightning =0.9.0
    pip install pytorch =1.8.2
    ...
    ```
