import os
import cv2
import glob
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips 
import matplotlib
matplotlib.use('Agg') # Force headless plotting
import matplotlib.pyplot as plt

# ==============================================================================
# BLOCK 1: CONFIGURATION & SETUP
# ==============================================================================
"""
METHODOLOGY: Reproducibility & Hardware Acceleration.
We set random seeds to ensure that if we run the experiment twice, we get the exact 
same results (critical for scientific validation). We also auto-detect GPU/CPU.
"""
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATASET_DIR = os.path.join(BASE_DIR, "dataset") 
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
HR_DIR = os.path.join(DATA_DIR, "HR")
LR_DIR = os.path.join(DATA_DIR, "LR")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for d in [HR_DIR, LR_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# VGG Normalization Constants (ImageNet)
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
VGG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

# ==============================================================================
# BLOCK 2: DATA PREPARATION & PRE-PROCESSING
# ==============================================================================
"""
METHODOLOGY: Domain-Specific Enhancement.
Retinal images are dominated by the Green Channel (where vessels are visible).
ALGORITHM: CLAHE (Contrast Limited Adaptive Histogram Equalization).
1. Split RGB.
2. Apply CLAHE to Green channel (ClipLimit=2.0, Grid=8x8).
3. Merge back. This pre-amplifies vessel signals for the network.
"""
def prepare_data():
    if len(os.listdir(HR_DIR)) > 0:
        print(">>> Data already processed. Skipping preparation.")
        return

    print(">>> Preparing Data...")
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(SOURCE_DATASET_DIR, ext)))

    if len(image_paths) == 0:
        print("!!! No data found. Creating dummy retinal noise data for demonstration...")
        dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(SOURCE_DATASET_DIR, "dummy_eye.jpg"), dummy)
        image_paths = [os.path.join(SOURCE_DATASET_DIR, "dummy_eye.jpg")]

    SCALE = 4
    MIN_SIZE = 256
    count = 0

    for path in tqdm(image_paths):
        img_name = os.path.basename(path)
        hr_img = cv2.imread(path)
        if hr_img is None: continue

        h, w, _ = hr_img.shape
        if h < MIN_SIZE or w < MIN_SIZE: continue

        # Crop to be divisible by scale
        h_new, w_new = h - (h % SCALE), w - (w % SCALE)
        hr_img = hr_img[:h_new, :w_new]
        
        # ALGORITHM: Bicubic Downsampling
        # We simulate Low-Res inputs using bicubic interpolation.
        lr_img = cv2.resize(hr_img, (w_new//SCALE, h_new//SCALE), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(HR_DIR, img_name), hr_img)
        cv2.imwrite(os.path.join(LR_DIR, img_name), lr_img)
        count += 1
    
    print(f"Data Prepared: {count} valid images.")

def apply_green_clahe(image_rgb):
    r, g, b = cv2.split(image_rgb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    return cv2.merge((r, g, b))

# ==============================================================================
# BLOCK 3: DISEASE-AWARE DATASET LOADER
# ==============================================================================
"""
METHODOLOGY: Hard Example Mining / Disease-Aware Sampling.
Random cropping misses small lesions (exudates/hemorrhages).
ALGORITHM: Color Thresholding & Contour Detection.
1. Calculate 98th percentile (Bright Spots) and 5th percentile (Dark Spots).
2. Threshold image to find these 'anomalies'.
3. 70% of batches are forced to center on these anomalies.
"""
class DiseaseAwareDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=48, scale=4, augment=True, disease_focus_prob=0.7):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        self.disease_focus_prob = disease_focus_prob 
        self.files = sorted(os.listdir(lr_dir))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def get_lesion_coordinates(self, img_rgb):
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_gray)
        
        # Algorithm: Statistical Thresholding
        thresh_bright = np.percentile(enhanced, 98)
        _, mask_bright = cv2.threshold(enhanced, thresh_bright, 255, cv2.THRESH_BINARY)
        
        thresh_dark = np.percentile(enhanced, 5)
        _, mask_dark = cv2.threshold(enhanced, thresh_dark, 255, cv2.THRESH_BINARY_INV)
        
        mask_combined = cv2.bitwise_or(mask_bright, mask_dark)
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 20: 
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    candidates.append((cx, cy))
        return candidates

    def __getitem__(self, idx):
        name = self.files[idx]
        lr = cv2.imread(os.path.join(self.lr_dir, name))
        hr = cv2.imread(os.path.join(self.hr_dir, name))
        
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        hr = apply_green_clahe(hr) 

        h, w, _ = lr.shape
        if self.patch_size == 0:
            return self.to_tensor(lr), self.to_tensor(hr)

        ps = min(self.patch_size, h, w)
        x, y = 0, 0
        found_lesion = False
        
        # Sampling Strategy: Bias towards lesions
        if random.random() < self.disease_focus_prob:
            candidates = self.get_lesion_coordinates(lr)
            if len(candidates) > 0:
                cx, cy = random.choice(candidates)
                x = max(0, min(cx - ps // 2, w - ps))
                y = max(0, min(cy - ps // 2, h - ps))
                found_lesion = True
        
        if not found_lesion:
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)

        lr_p = lr[y:y+ps, x:x+ps]
        hr_p = hr[y*self.scale:(y+ps)*self.scale, x*self.scale:(x+ps)*self.scale]

        if self.augment:
            if random.random() > 0.5:
                lr_p = cv2.flip(lr_p, 1); hr_p = cv2.flip(hr_p, 1)
            if random.random() > 0.5:
                lr_p = cv2.flip(lr_p, 0); hr_p = cv2.flip(hr_p, 0)
            k = random.randint(0, 3)
            if k > 0:
                lr_p = np.rot90(lr_p, k); hr_p = np.rot90(hr_p, k)

        lr_p = np.ascontiguousarray(lr_p)
        hr_p = np.ascontiguousarray(hr_p)
        return self.to_tensor(lr_p), self.to_tensor(hr_p)

# ==============================================================================
# BLOCK 4: NETWORK ARCHITECTURE (ESRGAN)
# ==============================================================================
"""
METHODOLOGY: Residual-in-Residual Dense Block (RRDB).
Standard SRGAN uses Batch Norm (BN), which creates artifacts in medical images.
We remove BN and use RRDB for deeper, more stable training.
ALGORITHM: 
1. Feature Extraction (Conv).
2. Non-Linear Mapping (RRDB Blocks).
3. Upsampling (Sub-pixel Convolution / PixelShuffle).
"""
class DenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2*gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3*gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4*gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.db1 = DenseBlock(nf); self.db2 = DenseBlock(nf); self.db3 = DenseBlock(nf)
    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return out * 0.2 + x

class ESRGAN_Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=64, nb=16):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.conv_trunk = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1); self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1); self.pixel_shuffle2 = nn.PixelShuffle(2)
        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.last_conv = nn.Conv2d(nf, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_trunk(self.rrdb_blocks(fea))
        fea = fea + trunk
        fea = self.lrelu(self.pixel_shuffle1(self.upconv1(fea)))
        fea = self.lrelu(self.pixel_shuffle2(self.upconv2(fea)))
        out = self.last_conv(self.lrelu(self.hr_conv(fea)))
        base = torch.nn.functional.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        return out + base 

class PatchGAN_Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def d_block(in_f, out_f, norm=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_f, out_f, 4, 2, 1))] 
            if norm: layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *d_block(in_channels, 64, False),
            *d_block(64, 128),
            *d_block(128, 256),
            *d_block(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1)
        )
    def forward(self, x): return self.model(x)

# ==============================================================================
# BLOCK 5: LOSS FUNCTIONS (THE "MEDICAL MIX")
# ==============================================================================
"""
METHODOLOGY: Texture & Edge Retention.
ALGORITHMS:
1. Sobel Edge Loss: Convolves image with Sobel kernels to minimize edge blurring.
2. Vessel-Aware L1: Penalizes Green channel errors 1.5x more.
3. VGG Perceptual: Minimizes distance in Feature Space (conv3_4), not Pixel Space.
"""
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3)
        self.register_buffer('k_x', k_x)
        self.register_buffer('k_y', k_y)
        self.loss = nn.L1Loss()

    def forward(self, sr, hr):
        sr_gray = sr.mean(dim=1, keepdim=True)
        hr_gray = hr.mean(dim=1, keepdim=True)
        sr_x = torch.nn.functional.conv2d(sr_gray, self.k_x, padding=1)
        sr_y = torch.nn.functional.conv2d(sr_gray, self.k_y, padding=1)
        hr_x = torch.nn.functional.conv2d(hr_gray, self.k_x, padding=1)
        hr_y = torch.nn.functional.conv2d(hr_gray, self.k_y, padding=1)
        return self.loss(sr_x, hr_x) + self.loss(sr_y, hr_y)

def vessel_aware_l1_loss(sr, hr, l1):
    base_loss = l1(sr, hr)
    sr_g, hr_g = sr[:, 1, :, :], hr[:, 1, :, :]
    return base_loss + 1.5 * torch.abs(sr_g - hr_g).mean()

# ==============================================================================
# BLOCK 6: TRAINING LOOP (TWO-STAGE)
# ==============================================================================
"""
METHODOLOGY: Curriculum Learning.
1. Warmup: Train only Generator with L1 Loss. Learn Structure.
2. GAN: Train G & D with Adversarial + Perceptual Loss. Learn Texture.
ALGORITHM: Gradient Descent with Adam Optimizer + Cosine Annealing.
"""
def train():
    BATCH_SIZE = 4
    PATCH_SIZE = 48 
    
    dataset = DiseaseAwareDataset(LR_DIR, HR_DIR, patch_size=PATCH_SIZE, augment=True, disease_focus_prob=0.7)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    print(f">>> Dataset Loaded: {len(dataset)} valid pairs.")
    
    G = ESRGAN_Generator().to(device)
    D = PatchGAN_Discriminator().to(device)
    edge_loss_fn = EdgeLoss().to(device)
    
    vgg = models.vgg19(pretrained=True).features[:35].to(device).eval()
    for p in vgg.parameters(): p.requires_grad = False
    
    opt_G = optim.Adam(G.parameters(), lr=1e-4)
    opt_D = optim.Adam(D.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    # --- STAGE 1: WARMUP ---
    WARMUP_EPOCHS = 5 
    print(f"\n{'='*10} STAGE 1: WARMUP (Structure) {'='*10}")
    
    for epoch in range(WARMUP_EPOCHS):
        loop = tqdm(loader, leave=True)
        for lr, hr in loop:
            if lr.size(0) == 0: continue
            lr, hr = lr.to(device), hr.to(device)
            
            opt_G.zero_grad()
            with torch.cuda.amp.autocast():
                sr = G(lr)
                loss = vessel_aware_l1_loss(sr, hr, l1) + 0.5 * edge_loss_fn(sr, hr)
            
            scaler.scale(loss).backward()
            scaler.step(opt_G)
            scaler.update()
            loop.set_description(f"Warmup {epoch+1}/{WARMUP_EPOCHS}")
            
    torch.save(G.state_dict(), os.path.join(MODEL_DIR, "G_warmup.pth"))

    # --- STAGE 2: GAN ---
    GAN_EPOCHS = 10 
    print(f"\n{'='*10} STAGE 2: GAN (Detail/Texture) {'='*10}")
    
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=GAN_EPOCHS, eta_min=1e-7)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=GAN_EPOCHS, eta_min=1e-7)

    for epoch in range(GAN_EPOCHS):
        loop = tqdm(loader, leave=True)
        for lr, hr in loop:
            if lr.size(0) == 0: continue
            lr, hr = lr.to(device), hr.to(device)
            
            # 1. Train Discriminator
            opt_D.zero_grad()
            with torch.cuda.amp.autocast():
                sr = G(lr).detach()
                real_pred = D(hr)
                fake_pred = D(sr)
                d_loss = (bce(real_pred, torch.ones_like(real_pred)) + 
                          bce(fake_pred, torch.zeros_like(fake_pred))) / 2
            scaler.scale(d_loss).backward()
            scaler.step(opt_D)
            scaler.update()

            # 2. Train Generator
            opt_G.zero_grad()
            with torch.cuda.amp.autocast():
                sr = G(lr)
                fake_pred = D(sr)
                sr_norm = (sr - VGG_MEAN) / VGG_STD
                hr_norm = (hr - VGG_MEAN) / VGG_STD
                
                # TEXTURE FOCUSED LOSS WEIGHTS
                perceptual = l1(vgg(sr_norm), vgg(hr_norm))
                adversarial = bce(fake_pred, torch.ones_like(fake_pred))
                content = vessel_aware_l1_loss(sr, hr, l1)
                edge = edge_loss_fn(sr, hr)
                
                # Weights: Low content, High Adv/Perc for Texture
                g_loss = 0.01 * content + 1.0 * perceptual + 0.02 * adversarial + 0.5 * edge
            
            scaler.scale(g_loss).backward()
            scaler.step(opt_G)
            scaler.update()
            loop.set_description(f"GAN {epoch+1}/{GAN_EPOCHS}")
        
        scheduler_G.step()
        scheduler_D.step()

    torch.save(G.state_dict(), os.path.join(MODEL_DIR, "G_final.pth"))
    print("Training Complete.")

# ==============================================================================
# BLOCK 7: INFERENCE, INTERPOLATION & METRICS
# ==============================================================================
"""
METHODOLOGY: Network Interpolation & TTA.
1. Interpolation: We blend the weights of the 'Sharp' GAN and 'Blurry' Warmup model.
   Alpha=0.2 means we keep 80% of the GAN's texture.
2. Geometric Ensemble: We rotate/flip the input 8 times and average the output.
   This cancels out random noise artifacts from the GAN.
"""
def net_interpolation(warmup_path, gan_path, alpha=0.2):
    net_psnr = torch.load(warmup_path)
    net_gan = torch.load(gan_path)
    net_interp = {}
    for k, v_psnr in net_psnr.items():
        v_gan = net_gan[k]
        net_interp[k] = (1 - alpha) * v_gan + alpha * v_psnr
    
    model = ESRGAN_Generator().to(device)
    model.load_state_dict(net_interp)
    return model

def geometric_ensemble(model, lr):
    lr_list = [
        lr, 
        torch.rot90(lr, 1, [2, 3]), torch.rot90(lr, 2, [2, 3]), torch.rot90(lr, 3, [2, 3]),
        torch.flip(lr, [3]),
        torch.rot90(torch.flip(lr, [3]), 1, [2, 3]),
        torch.rot90(torch.flip(lr, [3]), 2, [2, 3]),
        torch.rot90(torch.flip(lr, [3]), 3, [2, 3])
    ]
    sr_list = []
    for x in lr_list: sr_list.append(model(x))
    
    sr_list[1] = torch.rot90(sr_list[1], 3, [2, 3])
    sr_list[2] = torch.rot90(sr_list[2], 2, [2, 3])
    sr_list[3] = torch.rot90(sr_list[3], 1, [2, 3])
    sr_list[4] = torch.flip(sr_list[4], [3])
    sr_list[5] = torch.flip(torch.rot90(sr_list[5], 3, [2, 3]), [3])
    sr_list[6] = torch.flip(torch.rot90(sr_list[6], 2, [2, 3]), [3])
    sr_list[7] = torch.flip(torch.rot90(sr_list[7], 1, [2, 3]), [3])
    
    return torch.stack(sr_list).mean(dim=0)

def process_full_image(model):
    print("\n>>> Processing Full Image & Calculating Metrics...")
    
    # Initialize LPIPS 
    try:
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        print("LPIPS Loaded successfully.")
    except Exception as e:
        print(f"Warning: LPIPS could not be loaded ({e}). Install with 'pip install lpips'. Skipping LPIPS.")
        loss_fn_alex = None

    dataset = DiseaseAwareDataset(LR_DIR, HR_DIR, patch_size=0, augment=False)
    
    idx = random.randint(0, len(dataset)-1)
    lr_tensor, hr_tensor = dataset[idx]
    
    lr = lr_tensor.unsqueeze(0).to(device)
    hr = hr_tensor.unsqueeze(0).to(device)
    
    _, _, h, w = lr.shape
    print(f"    Input: {h}x{w} | Target: {h*4}x{w*4}")
    
    with torch.no_grad():
        sr = geometric_ensemble(model, lr)
        bic = torch.nn.functional.interpolate(lr, scale_factor=4, mode='bicubic')
        
        if loss_fn_alex:
            sr_norm = (sr - 0.5) * 2
            hr_norm = (hr - 0.5) * 2
            lpips_val = loss_fn_alex(sr_norm, hr_norm).item()
        else:
            lpips_val = 0.0

    def to_np(t): return t.squeeze().permute(1,2,0).cpu().numpy().clip(0,1)
    
    sr_img = to_np(sr)
    hr_img = to_np(hr)
    bic_img = to_np(bic)
    
    psnr_val = peak_signal_noise_ratio(hr_img, sr_img, data_range=1.0)
    ssim_val = structural_similarity(hr_img, sr_img, data_range=1.0, channel_axis=2)
    
    psnr_bic = peak_signal_noise_ratio(hr_img, bic_img, data_range=1.0)
    ssim_bic = structural_similarity(hr_img, bic_img, data_range=1.0, channel_axis=2)

    print(f"\n{'-'*30}")
    print(f"PERFORMANCE METRICS (Texture-Focused)")
    print(f"{'-'*30}")
    print(f"Metric   | Bicubic (Base) | Medical SRGAN (Ours)")
    print(f"---------|----------------|---------------------")
    print(f"PSNR     | {psnr_bic:.2f} dB       | {psnr_val:.2f} dB")
    print(f"SSIM     | {ssim_bic:.4f}         | {ssim_val:.4f}")
    print(f"LPIPS    | N/A            | {lpips_val:.4f} (Lower is better)")
    print(f"{'-'*30}\n")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes[0].imshow(bic_img)
    axes[0].set_title(f"Bicubic\nPSNR: {psnr_bic:.2f} | SSIM: {ssim_bic:.3f}", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(sr_img)
    axes[1].set_title(f"Medical SRGAN (Ours)\nPSNR: {psnr_val:.2f} | SSIM: {ssim_val:.3f} | LPIPS: {lpips_val:.3f}", 
                      fontsize=12, fontweight='bold', color='darkblue')
    axes[1].axis('off')
    
    axes[2].imshow(hr_img)
    axes[2].set_title("Ground Truth (HR)", fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "full_image_metrics.png")
    plt.savefig(out_path, dpi=200)
    print(f"SUCCESS: Comparison with metrics saved to {out_path}")
    plt.close(fig)

# ==============================================================================
# BLOCK 8: MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    prepare_data()
    
    if not os.path.exists(os.path.join(MODEL_DIR, "G_final.pth")):
        train()
    else:
        # Optional: Uncomment 'train()' to force retraining if you changed loss weights
        # train() 
        pass
    
    warmup_path = os.path.join(MODEL_DIR, "G_warmup.pth")
    gan_path = os.path.join(MODEL_DIR, "G_final.pth")
    
    print(f"\n>>> Interpolating Models (alpha=0.2 for TEXTURE priority)...")
    best_model = net_interpolation(warmup_path, gan_path, alpha=0.2)
    best_model.eval()
    
    process_full_image(best_model)
