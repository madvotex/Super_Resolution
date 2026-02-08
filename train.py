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

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
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

# ==========================================
# 2. DATA PREPARATION
# ==========================================
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
        raise FileNotFoundError(f"No images found in {SOURCE_DATASET_DIR}")

    SCALE = 4
    MIN_SIZE = 256
    count = 0

    for path in tqdm(image_paths):
        img_name = os.path.basename(path)
        hr_img = cv2.imread(path)
        if hr_img is None: continue

        h, w, _ = hr_img.shape
        if h < MIN_SIZE or w < MIN_SIZE: continue

        h_new, w_new = h - (h % SCALE), w - (w % SCALE)
        hr_img = hr_img[:h_new, :w_new]
        lr_img = cv2.resize(hr_img, (w_new//SCALE, h_new//SCALE), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(HR_DIR, img_name), hr_img)
        cv2.imwrite(os.path.join(LR_DIR, img_name), lr_img)
        count += 1
    
    print(f"Data Prepared: {count} valid images.")

# ==========================================
# 3. DATASET
# ==========================================
def apply_green_clahe(image_rgb):
    r, g, b = cv2.split(image_rgb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    return cv2.merge((r, g, b))

class RetinalDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=48, scale=4, augment=True):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        self.files = sorted(os.listdir(lr_dir))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

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
        for _ in range(10): 
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)
            lr_p = lr[y:y+ps, x:x+ps]
            if lr_p.mean() > 20: 
                break

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

# ==========================================
# 4. MODELS & LOSSES
# ==========================================
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
            layers = [nn.Conv2d(in_f, out_f, 4, 2, 1)]
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

# --- UPGRADE 1: EDGE LOSS (Sobel) ---
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

# ==========================================
# 5. TRAINING
# ==========================================
def train():
    BATCH_SIZE = 4
    PATCH_SIZE = 48 
    
    dataset = RetinalDataset(LR_DIR, HR_DIR, patch_size=PATCH_SIZE, augment=True)
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
    WARMUP_EPOCHS = 30
    print(f"\n{'='*10} STAGE 1: WARMUP (30 Epochs) {'='*10}")
    for epoch in range(WARMUP_EPOCHS):
        loop = tqdm(loader, leave=True)
        for lr, hr in loop:
            if lr.size(0) == 0: continue
            lr, hr = lr.to(device), hr.to(device)
            
            opt_G.zero_grad()
            with torch.cuda.amp.autocast():
                sr = G(lr)
                # Combined Pixel + Edge Loss for sharp warmup
                loss = vessel_aware_l1_loss(sr, hr, l1) + 0.5 * edge_loss_fn(sr, hr)
            
            scaler.scale(loss).backward()
            scaler.step(opt_G)
            scaler.update()
            loop.set_description(f"Warmup {epoch+1}")
            
    torch.save(G.state_dict(), os.path.join(MODEL_DIR, "G_warmup.pth"))

    # --- STAGE 2: GAN ---
    GAN_EPOCHS = 30
    print(f"\n{'='*10} STAGE 2: GAN (30 Epochs) {'='*10}")
    
    # --- UPGRADE 2: COSINE SCHEDULER ---
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=GAN_EPOCHS, eta_min=1e-7)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=GAN_EPOCHS, eta_min=1e-7)

    for epoch in range(GAN_EPOCHS):
        loop = tqdm(loader, leave=True)
        for lr, hr in loop:
            if lr.size(0) == 0: continue
            lr, hr = lr.to(device), hr.to(device)
            
            # Train D
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

            # Train G
            opt_G.zero_grad()
            with torch.cuda.amp.autocast():
                sr = G(lr)
                fake_pred = D(sr)
                adv = bce(fake_pred, torch.ones_like(fake_pred))
                content = vessel_aware_l1_loss(sr, hr, l1)
                
                # Safe VGG
                if sr.shape[-1] < 32: sr_v = torch.nn.functional.interpolate(sr, (32,32))
                elif sr.shape[-1] > 224: sr_v = torch.nn.functional.interpolate(sr, (224,224))
                else: sr_v = sr
                
                if hr.shape[-1] < 32: hr_v = torch.nn.functional.interpolate(hr, (32,32))
                elif hr.shape[-1] > 224: hr_v = torch.nn.functional.interpolate(hr, (224,224))
                else: hr_v = hr
                
                perc = l1(vgg(sr_v), vgg(hr_v))
                edge = edge_loss_fn(sr, hr)
                
                # TOTAL LOSS (Content=0.05, Perc=1.0, Adv=0.005, Edge=0.5)
                g_loss = 0.05 * content + 1.0 * perc + 0.005 * adv + 0.5 * edge
            
            scaler.scale(g_loss).backward()
            scaler.step(opt_G)
            scaler.update()
            loop.set_description(f"GAN {epoch+1}")
        
        # Step schedulers at end of epoch
        scheduler_G.step()
        scheduler_D.step()

    torch.save(G.state_dict(), os.path.join(MODEL_DIR, "G_final.pth"))
    print("Training Complete.")
    return G

# ==========================================
# 6. INTERPOLATION & EVALUATION
# ==========================================
def net_interpolation(warmup_path, gan_path, alpha=0.7):
    print(f"\n>>> Interpolating Models (alpha={alpha})...")
    net_psnr = torch.load(warmup_path)
    net_gan = torch.load(gan_path)
    net_interp = {}
    for k, v_psnr in net_psnr.items():
        v_gan = net_gan[k]
        net_interp[k] = (1 - alpha) * v_gan + alpha * v_psnr
    
    save_path = os.path.join(MODEL_DIR, "G_interpolated.pth")
    torch.save(net_interp, save_path)
    
    model = ESRGAN_Generator().to(device)
    model.load_state_dict(net_interp)
    return model

# --- UPGRADE 3: FULL 8x ENSEMBLE ---
def geometric_ensemble(model, lr):
    # Generates 8 geometric variations
    lr_list = [
        lr, 
        torch.rot90(lr, 1, [2, 3]),
        torch.rot90(lr, 2, [2, 3]),
        torch.rot90(lr, 3, [2, 3]),
        torch.flip(lr, [3]),
        torch.rot90(torch.flip(lr, [3]), 1, [2, 3]),
        torch.rot90(torch.flip(lr, [3]), 2, [2, 3]),
        torch.rot90(torch.flip(lr, [3]), 3, [2, 3])
    ]
    
    sr_list = []
    for x in lr_list:
        sr_list.append(model(x))
    
    # Inverse transforms
    sr_list[1] = torch.rot90(sr_list[1], 3, [2, 3])
    sr_list[2] = torch.rot90(sr_list[2], 2, [2, 3])
    sr_list[3] = torch.rot90(sr_list[3], 1, [2, 3])
    sr_list[4] = torch.flip(sr_list[4], [3])
    sr_list[5] = torch.flip(torch.rot90(sr_list[5], 3, [2, 3]), [3])
    sr_list[6] = torch.flip(torch.rot90(sr_list[6], 2, [2, 3]), [3])
    sr_list[7] = torch.flip(torch.rot90(sr_list[7], 1, [2, 3]), [3])
    
    return torch.stack(sr_list).mean(dim=0)

def evaluate_and_plot(model):
    print("\n>>> Visualizing Results & Calculating Metrics...")
    dataset = RetinalDataset(LR_DIR, HR_DIR, patch_size=0, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    try:
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    except Exception as e:
        print(f"Error loading LPIPS: {e}"); return

    model.eval()
    try: lr, hr = next(iter(loader))
    except StopIteration: print("No data."); return

    lr, hr = lr.to(device), hr.to(device)
    
    print("   > Running 8x Ensemble Inference...")
    with torch.no_grad():
        sr = geometric_ensemble(model, lr)
        bic = torch.nn.functional.interpolate(lr, scale_factor=4, mode='bicubic')
        
        sr_norm = (sr - 0.5) * 2; hr_norm = (hr - 0.5) * 2
        lpips_val = loss_fn_alex(sr_norm, hr_norm).item()
        
    def to_np(t): return t.squeeze().permute(1,2,0).cpu().numpy().clip(0,1)
    
    sr_np, hr_np, bic_np = to_np(sr), to_np(hr), to_np(bic)
    psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
    ssim = structural_similarity(hr_np, sr_np, data_range=1.0, channel_axis=2)

    print(f"\n{'-'*30}")
    print(f"FINAL EVALUATION RESULTS (Enhanced Model):")
    print(f"{'-'*30}")
    print(f"PSNR : {psnr:.2f} dB")
    print(f"SSIM : {ssim:.4f}")
    print(f"LPIPS: {lpips_val:.4f}")
    print(f"{'-'*30}\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(bic_np); axes[0].set_title("Before (Bicubic)"); axes[0].axis('off')
    
    axes[1].imshow(sr_np)
    axes[1].set_title(f"After (Medical Grade)\nPSNR: {psnr:.2f} | SSIM: {ssim:.3f} | LPIPS: {lpips_val:.3f}", 
                      fontsize=11, fontweight='bold', color='darkblue')
    axes[1].axis('off')
    
    axes[2].imshow(hr_np); axes[2].set_title("Ground Truth"); axes[2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "result_final.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"SUCCESS: Result saved to {save_path}")

if __name__ == "__main__":
    prepare_data()
    train() 
    
    warmup_path = os.path.join(MODEL_DIR, "G_warmup.pth")
    gan_path = os.path.join(MODEL_DIR, "G_final.pth")
    
    # alpha=0.7: High Structure retention
    best_model = net_interpolation(warmup_path, gan_path, alpha=0.7)
    
    evaluate_and_plot(best_model)