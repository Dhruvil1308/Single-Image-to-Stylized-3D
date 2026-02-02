import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm

# Configuration
MODEL_NAME = "Lykon/AnyLoRA"
DATA_DIR = "data/processed/imfdb"
OUTPUT_DIR = "models/indianface_lora"
RESOLUTION = 512
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
MAX_TRAIN_STEPS = 500  # Initial fast training
LORA_R = 8
LORA_ALPHA = 16

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Force NVIDIA GPU if available
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda")
        print(f"CUDA detected. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA NOT DETECTED. Training on CPU will be extremely slow.")
        device = torch.device("cpu")

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
    
    # Move to device immediately to free CPU RAM
    text_encoder.to(device)
    pipeline.vae.to(device)
    
    # Freeze background components
    text_encoder.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    
    # 2. Configure LoRA
    print("2/6: Configuring LoRA layers...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    
    # 4GB VRAM Optimizations
    print("3/6: Applying VRAM optimizations...")
    unet.enable_gradient_checkpointing()
    # unet.enable_xformers_memory_efficient_attention() # Disabled for Windows stability

    # 8-bit optimization
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=LEARNING_RATE)
        print("Using 8-bit Adam for 4GB VRAM.")
    except Exception as e:
        print(f"Bitsandbytes fallback: {e}")
        optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    # 3. Dataset Preparation
    print("4/6: Preparing dataset...")
    train_transforms = transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess_train(examples):
        images = [train_transforms(image.convert("RGB")) for image in examples["image"]]
        captions = examples["text"]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {"pixel_values": images, "input_ids": inputs.input_ids}

    dataset = load_dataset("imagefolder", data_dir=DATA_DIR, split="train")
    train_dataset = dataset.with_transform(preprocess_train)
    
    # On Windows, num_workers > 0 often causes hangs in data loading
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)

    # 4. Accelerator Setup
    print("5/6: Preparing accelerator...")
    unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)
    
    # 5. Training Loop
    print("6/6: Starting training loop...")
    global_step = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), desc="Steps")
    
    unet.train()
    for epoch in range(100):
        for batch in train_loader:
            with accelerator.accumulate(unet):
                # We need to move pixel values to device manually if not fully handled by accelerator
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
                input_ids = batch["input_ids"].to(device)
                
                # Sample noise
                latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embedding
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Compute loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step >= MAX_TRAIN_STEPS:
                    break
        if global_step >= MAX_TRAIN_STEPS:
            break

    # 6. Save LoRA weights
    print(f"All done! Saving weights to {OUTPUT_DIR}...")
    unet.save_pretrained(OUTPUT_DIR)
    print("Success. Model ready for Indian Avatar generation.")

if __name__ == "__main__":
    train()
