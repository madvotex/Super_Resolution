Medical ESRGAN: Clinical-Grade Retinal Image Super-Resolution
This repository contains a highly specialized adaptation of the Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) designed specifically for medical imaging, with a focus on retinal fundus photography.

Unlike standard super-resolution models that optimize for photorealistic textures (often resulting in dangerous clinical hallucinations like fake lesions), this architecture strictly optimizes for mathematical structural integrity, capillary preservation, and geometric accuracy.

Overview
The goal of this project is to perform a 4x upscale on low-resolution retinal images while maintaining a Structural Similarity Index (SSIM) suitable for clinical diagnostics. To achieve this, the standard SRGAN pipeline has been heavily modified with curriculum learning, disease-aware sampling, and a custom multi-objective loss function.

Key Clinical Features
Disease-Aware Hard Example Mining: Retinal images are predominantly empty background. This model uses a custom dataloader featuring CLAHE (Contrast Limited Adaptive Histogram Equalization) and statistical thresholding to force 70% of training batches to center on clinical anomalies (exudates, hemorrhages, or microaneurysms).

Green-Channel Vessel Penalty: Blood vessels absorb green light, making them most visible in the green channel of an RGB image. The loss function applies a 1.5x penalty to errors in the green channel, forcing the network to prioritize vascular sharpness.

Multi-Scale SSIM Hybrid Loss: Replaces standard MSE with a hybrid loss utilizing L1 (pixel intensity), MS-SSIM (structural neighborhood mapping), and Sobel edge detection to prevent the blurring of the optic disc and vessel boundaries.

Curriculum Learning (Pure L1 Warmup): The generator undergoes an extended 15-epoch warmup using pure L1 loss to learn exact structural mappings before the discriminator is activated, preventing early-stage color corruption.

Network Interpolation: The final inference model utilizes weight interpolation (alpha blending). By keeping 80% of the structurally safe L1 warmup weights and only 20% of the GAN's texture weights, the model safely navigates the perception-distortion tradeoff.

Dynamic Fundus Masking: Evaluation metrics automatically mask out the massive black borders of fundus images, ensuring PSNR and SSIM scores reflect the true accuracy of the biological tissue, not the empty background.

Requirements
Python 3.8+

PyTorch (with CUDA support recommended)

OpenCV (cv2)

scikit-image

LPIPS (pip install lpips)

tqdm

matplotlib

Directory Structure
Upon running the script, the following directory structure will be generated automatically:

Plaintext

├── dataset/               # Place your raw high-resolution retinal images here
├── processed_data/        # Auto-generated HR and downsampled LR pairs
│   ├── HR/
│   └── LR/
├── models/                # Saved Generator and Discriminator weights (.pth)
├── outputs/               # Visual metric comparisons and evaluation plots
└── main.py                # The primary execution script
Usage
Prepare Data: Place your original retinal images (jpg, png, tif) into the dataset/ directory. If the folder is empty, the script will generate a dummy image for demonstration purposes.

Execute Pipeline: Run the main script. The script automatically handles data pre-processing (bicubic downsampling), Stage 1 Warmup training, Stage 2 GAN training, and final interpolation.

Bash

python main.py
Evaluate: Once training is complete, check the outputs/ folder for full_image_metrics_interpolated.png to view a side-by-side comparison of Bicubic vs. Medical SRGAN vs. Ground Truth, along with the masked PSNR, SSIM, and LPIPS metrics.

Architecture Details
Generator: Residual-in-Residual Dense Blocks (RRDB) with Batch Normalization explicitly removed to prevent batch-statistic color shifting in clinical tissues.

Discriminator: PatchGAN discriminator utilizing spectral normalization to stabilize adversarial training.

Adversarial Weighting: The adversarial loss weight is intentionally crippled (0.001) to act as a mild texture sharpener rather than a hallucination engine.
