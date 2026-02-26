import torch
from diffusers import StableDiffusionPipeline
import os

def test_lora():
    model_id = "Lykon/AnyLoRA"
    lora_path = "models/indianface_lora"
    output_image = "assets/final_validation_portrait.png"
    
    print(f"Loading base model {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    
    print(f"Loading our Deep Accuracy LoRA: {lora_path}...")
    pipe.load_lora_weights(lora_path)
    
    prompt = "a professional 8k front-facing portrait of a beautiful indian person, indianface style, high quality, symmetrical facial features, masterpiece"
    negative_prompt = "blurry, low quality, distorted, western features, deformed"
    
    print(f"Generating perfect result for prompt: {prompt}")
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=30, 
        guidance_scale=7.5
    ).images[0]
    
    os.makedirs("assets", exist_ok=True)
    image.save(output_image)
    print(f"Validation image saved to {output_image}")

if __name__ == "__main__":
    test_lora()
