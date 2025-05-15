# DeepFake Detection Video (dfdVid) Setup Guide

This guide will help you set up the environment for the DeepFake Detection Video project using Miniconda.

## Prerequisites

- Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.
- If you don't have Miniconda installed, you can install it using the following command:

```sh
winget install -e --id Anaconda.Miniconda3 --override "/AddToPath=1"
```

## Step-by-Step Setup

### 1. Initialize Miniconda

Initialize Miniconda for all shells:

```sh
conda init --all
```

### 2. Create a New Conda Environment

Create a new Conda environment named `dfdVid` with Python 3.10:

```sh
conda create --name dfdVid python=3.9.20
```

### 3. Activate the Conda Environment

Activate the newly created Conda environment:

```sh
conda activate dfdVid
```

### 4. Optional: Install CUDA Accelerator

If you have an NVIDIA GPU and want to use CUDA for acceleration, install the CUDA runtime and cuDNN:

```sh
conda install conda-forge::cuda-runtime=12.4.1 conda-forge::cudnn=9.2.1.18
```

### 5. Install Project Requirements

Install the required Python packages using `pip`:

```sh
pip install -r requirements.txt
```

### 6. Install Model

[Link](https://drive.google.com/file/d/1sRPxBUyenhNVFhzfh4pDOpH5sK-IK2IE/view?usp=sharing)

Place the model in the models folder

## Additional Information

- Ensure that your GPU drivers are up to date if you are using CUDA.
- For more information on using Conda, refer to the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/index.html).

## Running the Project

After setting up the environment, you can run the project scripts as needed. For example:

```sh
python inference.py
```

## Troubleshooting

If you encounter any issues during setup, consider the following steps:

- Ensure that Miniconda is correctly installed and added to your system's PATH.
- Verify that the Conda environment is activated before running any commands.
- Check the compatibility of the installed packages with your system and Python version.

For further assistance, refer to the project's documentation or seek help from the community.
