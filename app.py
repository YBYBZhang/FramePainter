import os
import time

import torch
import gradio as gr
import spaces

from PIL import Image
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from safetensors.torch import load_file

from pipelines.pipeline_framepainter import FramePainterPipeline
from modules.sparse_control_encoder import SparseControlEncoder
from modules.unet_spatio_temporal_condition_edit import UNetSpatioTemporalConditionEdit
from modules.attention_processors import MatchingAttnProcessor2_0
from utils.attention_utils import set_matching_attention, set_matching_attention_processor



class timer:
    def __init__(self, method_name="timed process"):
        self.method = method_name

    def __enter__(self):
        self.start = time.time()
        print(f"{self.method} starts")

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        print(f"{self.method} took {str(round(end - self.start, 2))}s")


js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

pretrained_model_name_or_path = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
framepainter_path = "./checkpoints/FramePainter"

width = 1024
height = 576

unet = UNetSpatioTemporalConditionEdit.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="unet",
    low_cpu_mem_usage=True,
)
sparse_control_encoder = SparseControlEncoder()


image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    pretrained_model_name_or_path, subfolder="image_encoder")
vae = AutoencoderKLTemporalDecoder.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae")
noise_scheduler = EulerDiscreteScheduler.from_pretrained(
    pretrained_model_name_or_path, subfolder="scheduler")
pipeline = FramePainterPipeline.from_pretrained(
    pretrained_model_name_or_path,
    sparse_control_encoder=sparse_control_encoder, 
    unet=unet,
    vae=vae,
    image_encoder=image_encoder,
    revision=None,
    noise_scheduler=noise_scheduler
    )

set_matching_attention(pipeline.unet)
set_matching_attention_processor(pipeline.unet, MatchingAttnProcessor2_0(batch_size=2))


pipeline.set_progress_bar_config(disable=False)
pipeline.sparse_control_encoder.load_state_dict(load_file(os.path.join(framepainter_path, "encoder_diffusion_pytorch_model.safetensors")), strict=True)
pipeline.unet.load_state_dict(load_file(os.path.join(framepainter_path, "unet_diffusion_pytorch_model.safetensors")), strict=True)
pipeline.to("cuda")

header = """
<h1 align="left">FramePainter: Endowing Interactive Image Editing with Video Diffusion Priors</h1>
<div style="text-align: center; display: flex; justify-content: left; gap: 5px;">
<a href="https://arxiv.org/abs/2501.08225" align="center"><img src="https://img.shields.io/badge/ariXv-Paper-A42C25.svg" alt="arXiv"></a>
<a href="https://github.com/YBYBZhang/FramePainter" align="center"><img src="https://img.shields.io/badge/GitHub-Code-blue.svg?logo=github&" alt="GitHub"></a>
</div>
"""

with gr.Blocks(js=js_func) as demo:
    gr.HTML(header)
    with gr.Column():
        with gr.Row():
            with gr.Column():
                scribble = gr.ImageEditor(label="Input",type="pil", image_mode="RGB", sources='upload', brush=gr.Brush(default_size="6",color_mode="fixed", colors=["#FFFFFF"]), canvas_size=(1024, 576), height=600)
                steps = gr.Slider(label="Inference Steps", minimum=15, maximum=30, step=1, value=25, interactive=True)
                control_scale = gr.Slider(label="Control Scale", minimum=0.0, maximum=1.0, step=0.05, value=0.8, interactive=True)
                with gr.Row():
                    seed = gr.Number(label="Seed", value=3413, interactive=True)
                    btn = gr.Button(value="run")

            with gr.Column():
                output = gr.ImageEditor(label="Output", type="pil", canvas_size=(1024, 576))

        @spaces.GPU
        def process_image(steps, control_scale, seed, scribble):
            global pipeline
            if scribble is not None:   
                cond_image = [img.convert("L") for img in scribble['layers']] 
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16), timer("inference"):
                    scribble_resized = [img.resize((width, height)) for img in cond_image]
                    merged_image = Image.new("RGB", (width, height), (0,0,0))
                    
                    for layer in scribble_resized:
                        merged_image.paste(layer, (0, 0))
                    
                    input_image_resized = scribble["background"]
                    
                    validation_control_images = [
                        Image.new("RGB", (width, height), color=(0, 0, 0)), 
                        merged_image
                    ]
                    result = pipeline(
                        input_image_resized, 
                        validation_control_images,
                        height=height,
                        width=width,
                        edit_cond_scale=control_scale,
                        guidance_scale=3.0,
                        num_inference_steps=steps,
                        generator=torch.Generator().manual_seed(seed),
                    ).frames[0],
                    return result[0][1]
            else:
                return None
        reactive_controls = [steps, control_scale, seed, scribble]

        btn.click(process_image, inputs=reactive_controls, outputs=[output], show_progress="full")

# Main script
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)