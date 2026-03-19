## Medical ESRGAN: Clinical-Grade Retinal Image Super-Resolution

A domain-specific adaptation of ESRGAN designed for clinically safe retinal image super-resolution, prioritizing structural fidelity over perceptual realism.

---

## Overview

This project implements a 4× super-resolution pipeline for retinal fundus images while ensuring diagnostic reliability. Unlike conventional SRGAN/ESRGAN models that may introduce hallucinated features, this architecture enforces:

- Geometric consistency

- Capillary and vessel preservation

- Clinically meaningful reconstruction

The model is optimized for high Structural Similarity Index (SSIM) and low distortion, making it suitable for medical analysis workflows.

---

## Motivation

Standard super-resolution models optimize for visual appeal, which can lead to:

- Fake lesions

- Artificial textures

- Diagnostic misinterpretation

This project addresses that by enforcing mathematical and structural correctness instead of perceptual sharpness.

---

## Key Clinical Features
- Disease-Aware Hard Example Mining

- Uses CLAHE and statistical thresholding

- Forces 70% of batches to focus on pathological regions:

    - Microaneurysms

    - Hemorrhages

## Exudates

- Green-Channel Vessel Penalty

- Blood vessels are most visible in the green channel

- Loss function applies 1.5× weight to green-channel errors

- Enhances vascular clarity and continuity

## Multi-Scale SSIM Hybrid Loss

Replaces standard MSE with a composite loss:

- L1 Loss for pixel-level accuracy

- MS-SSIM for structural consistency

- Sobel Edge Loss for boundary preservation

Prevents:

- Blurring of optic disc

- Loss of vessel edges

## Curriculum Learning (L1 Warmup)

- First 15 epochs use pure L1 training

- GAN is activated only after stable structure learning

- Prevents hallucinations and color instability

## Network Interpolation
`Final Weights = 0.8 × L1 Model + 0.2 × GAN Model`

- Maintains structural safety

- Adds controlled texture enhancement

## Dynamic Fundus Masking

- Masks black background during evaluation

- Ensures metrics reflect actual retinal tissue quality

---

## Architecture
**Generator**

- Residual-in-Residual Dense Blocks (RRDB)

- Batch Normalization removed to prevent color shifts

**Discriminator**

- PatchGAN architecture

- Spectral normalization for stability

**Adversarial Loss**

- Weight = 0.001

- Acts as a refinement mechanism rather than a hallucination driver

---

Project Structure
- dataset  `  Raw high-resolution retinal images`
- processed_data       `  Generated HR/LR pairs`
  - HR
  - LR
-  Models                `  Saved model weights (.pth)`
- outputs               `  Evaluation results and comparisons`
- main.py                `  Main pipeline script`

---

## Requirements

- Python 3.8+

- PyTorch (CUDA recommended)

- OpenCV (cv2)

- scikit-image

- matplotlib

- tqdm

- LPIPS

**Install dependencies:**

`pip install torch torchvision opencv-python scikit-image matplotlib tqdm lpips`

---

## Usage
**Prepare Dataset**

Place retinal images in:

`dataset/`

Supported formats:

- .jpg

- .png

- .tif

If empty, a dummy image is automatically generated.

---

## Run Training Pipeline
`python main.py`

**Pipeline includes:**

- Data preprocessing (bicubic downsampling)

- Stage 1: L1 warmup training

- Stage 2: GAN training

- Model interpolation

- Evaluate Results

**Check:**

`outputs/full_image_metrics_interpolated.png`

**Includes:**

Bicubic vs Model vs Ground Truth

**Metrics:**

- PSNR

- SSIM

- LPIPS

## Evaluation Metrics

- PSNR for signal fidelity

- SSIM for structural similarity (primary metric)

- LPIPS for perceptual difference

All metrics are computed with fundus masking.

## Clinical Disclaimer

This model is designed to assist research and analysis, not replace medical diagnosis.

- Hallucination risk is minimized but not eliminated

- Requires validation before clinical deployment

- Not approved by regulatory authorities
