from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    custom_pipeline="hyoungwoncho/sd_perturbed_attention_guidance",
    torch_dtype=torch.float16,
    safety_checker=None
)

device="cuda"
pipe = pipe.to(device)

prompts = ["a corgi"]

output = pipe(
        prompts,
        width=512,
        height=512,
        num_inference_steps=50,
        guidance_scale=0.0,
        pag_scale=5.0,
        pag_applied_layers_index=['m0']
    ).images
