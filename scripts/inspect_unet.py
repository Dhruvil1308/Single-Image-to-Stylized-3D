import torch
from diffusers import UNet2DConditionModel

def inspect_unet():
    model_id = "Lykon/AnyLoRA"
    print(f"Loading UNet from {model_id}...")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)
    
    # Count parameters
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"Total UNet parameters: {total_params / 1e6:.2f}M")
    
    # Track modules
    print("\nSample module names (first 20):")
    for i, (name, module) in enumerate(unet.named_modules()):
        if i < 20:
            print(f"- {name}")
        
    # Check for specific targets
    targets = ["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]
    print("\nChecking target modules:")
    for target in targets:
        found = any(target in name for name, _ in unet.named_modules())
        print(f"Target '{target}': {'FOUND' if found else 'NOT FOUND'}")

if __name__ == "__main__":
    inspect_unet()
