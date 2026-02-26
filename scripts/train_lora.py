import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import json
import random
from PIL import Image
from torch.utils.data import Dataset

# Configuration
MODEL_NAME = "Lykon/AnyLoRA"
DATA_DIR = "data/processed/imfdb"
OUTPUT_DIR = "models/indianface_lora"
RESOLUTION = 512
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5 # Lowered for stability with more layers
MAX_TRAIN_STEPS = 2500 # Deep Accuracy Phase (~2 hours)
LORA_R = 4
LORA_ALPHA = 8

class IndianFaceDataset(Dataset):
    def __init__(self, data_dir, resolution=512, tokenizer=None):
        self.data_dir = data_dir
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.metadata_path = os.path.join(data_dir, "metadata.jsonl")
        self.entries = []
        
        with open(self.metadata_path, "r") as f:
            for line in f:
                self.entries.append(json.loads(line))
        
        # Random Sampling for Deep Accuracy
        random.shuffle(self.entries)
        self.entries = self.entries[:10000]
        print(f"Randomly loaded {len(self.entries)} diverse images for Deep Accuracy training.")

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.data_dir, entry["file_name"])
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            
            caption = entry["text"]
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids[0]
            
            return {"pixel_values": image, "input_ids": input_ids}
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__(0)

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Force NVIDIA GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="fp16",
    )

    # 1. Load Model
    print(f"1/6: Loading base model {MODEL_NAME}...")
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    
    # Enable VRAM optimizations
    pipeline.vae.to(device)
    pipeline.vae.enable_tiling()
    pipeline.vae.enable_slicing()
    text_encoder.to(device)
    text_encoder.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    
    # 2. Configure LoRA
    print("2/6: Configuring LoRA layers (Optimized Expanded Layers)...")
    target_modules = ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=0.0, # Set to zero for stability
        bias="none",
    )

    # Check for existing checkpoints
    latest_checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
            print(f"Found existing checkpoint: {latest_checkpoint}")

    if latest_checkpoint:
        try:
            print(f"Attempting to resume from {latest_checkpoint}...")
            unet = get_peft_model(unet, lora_config)
            state_dict = torch.load(os.path.join(latest_checkpoint, "adapter_model.bin"), map_location=device)
            set_peft_model_state_dict(unet, state_dict)
            print("Successfully resumed LoRA weights.")
        except Exception as e:
            print(f"Starting fresh for new accuracy layers: {e}")
            unet = get_peft_model(unet, lora_config)
            latest_checkpoint = None
    else:
        unet = get_peft_model(unet, lora_config)
    
    unet.enable_gradient_checkpointing()
    
    # Optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=LEARNING_RATE)
        print("Using 8-bit Adam for 4GB VRAM.")
    except ImportError:
        optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    # 4. Dataset
    train_dataset = IndianFaceDataset(DATA_DIR, RESOLUTION, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    # 5. Prepare
    unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)
    
    # 6. Loop
    global_step = 0
    if latest_checkpoint:
        global_step = int(os.path.basename(latest_checkpoint).split("-")[1])
        print(f"Resuming at step {global_step}")

    progress_bar = tqdm(range(MAX_TRAIN_STEPS), desc="Steps", initial=global_step)
    
    unet.train()
    for epoch in range(100):
        for step, batch in enumerate(train_loader):
            if global_step >= MAX_TRAIN_STEPS: break
            
            with accelerator.accumulate(unet):
                torch.cuda.empty_cache()
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
                input_ids = batch["input_ids"].to(device)
                
                with torch.no_grad():
                    latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * 0.18215
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
                
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]
                
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                if torch.isnan(loss):
                    print("NaN detected, skipping...")
                    optimizer.zero_grad()
                    continue

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % 100 == 0:
                    checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    unet.save_pretrained(checkpoint_dir)
                    print(f"Saved checkpoint: {checkpoint_dir}")
        
        if global_step >= MAX_TRAIN_STEPS: break

    unet.save_pretrained(OUTPUT_DIR)
    print("Success. Model ready.")

if __name__ == "__main__":
    train()
