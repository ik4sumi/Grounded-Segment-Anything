import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

from huggingface_hub import hf_hub_download

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler, ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import accelerate

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import requests
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline

import streamlit as st
from stqdm import stqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Grounded SAM
def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

# detect object using grounding DINO
def detect(image, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25, image_source=None):
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB
    return annotated_frame, boxes

def segment(image, sam_model, boxes):
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
        )
    return masks.cpu()

def process(local_image_path,prompt_sam,prompt_sdxl,refiner):

    print(local_image_path)

    model_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model_checkpoint_path = "groundingdino_swint_ogc.pth"

    groundingdino_model = load_model(model_config_path, model_checkpoint_path, device)

    sam_checkpoint = '/root/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    local_image_path = local_image_path

    image_source, image = load_image(local_image_path)
    Image.fromarray(image_source)

    annotated_frame, detected_boxes = detect(image, text_prompt=prompt_sam, model=groundingdino_model,image_source=image_source)
    segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)
    mask = segmented_frame_masks[0][0].cpu().numpy()
    inverted_mask = ((1 - mask) * 255).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)

    if not prompt_sdxl:
        prompt_sdxl='poster, phone, c4d,oc renderer, bright and cheerful, high saturation color, natural light, UI illustration, surrealism, rich in detail'
    #image = cv2.imread(local_image_path, cv2.IMREAD_COLOR)
    np_image = np.array(image_source)

    # get canny image
    np_canny_image = cv2.Canny(np_image, 100, 200)
    np_canny_image = np_canny_image[:, :, None]
    np_canny_image = np.concatenate([np_canny_image, np_canny_image, np_canny_image], axis=2)
    canny_image = Image.fromarray(np_canny_image)

    '''
    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # generate image
    generator = torch.manual_seed(123456)
    image = pipe(
        prompt,
        num_inference_steps=20,
        generator=generator,
        image=image,
        control_image=canny_image,
    ).images[0]
    '''
    controlnet = '/root/autodl-tmp/controlnet-canny-sdxl-1.0'
    vae = '/root/autodl-tmp/sdxl_vae.safetensors'
    sdxl = '/root/autodl-tmp/sd_xl_base_1.0.safetensors'

    controlnet_conditioning_scale = 0.5  # recommended for good generalization

    #vae = AutoencoderKL.from_single_file(vae)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,cache_dir='/root/autodl-tmp')

    controlnet = ControlNetModel.from_pretrained(
        controlnet,
        torch_dtype=torch.float16,
        revision="fp16"
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        cache_dir='/root/autodl-tmp'
    )
    pipe.enable_model_cpu_offload()
    generated_images = pipe(
        prompt_sdxl, negative_prompt='bad quality', image=canny_image, controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images

    #generated_images[0].save(f"hug_lab.png")

    # Copy masked image use inverted_mask on image
    #np_image = np.array(image)
    #np_image = np_image * (inverted_mask[:, :, None] /255)

    # 获取生成图像的尺寸
    generated_image = generated_images[0]
    gen_width, gen_height = generated_image.size

    # 如果掩码和生成图像的尺寸不同，则调整掩码的尺寸
    if mask.shape[:2] != (gen_height, gen_width):
        resized_mask = Image.fromarray(mask).resize((gen_width, gen_height), Image.Resampling.LANCZOS)
        resized_mask = np.array(resized_mask)
    else:
        resized_mask = inverted_mask

    # 确保 np_image 和 resized_mask 的形状相同
    if np_image.shape[:2] != resized_mask.shape[:2]:
        # 调整 np_image 的尺寸
        np_image = Image.fromarray(np_image).resize((gen_width, gen_height), Image.Resampling.LANCZOS)
        np_image = np.array(np_image)

    # 应用调整后的掩码到原始图像
    masked_np_image = np_image * (resized_mask[:, :, None] / 255)

    # 将掩码部分的原始图像叠加到生成的图像上
    # 先将生成图像转换为 NumPy 数组
    composite_image = np.array(generated_image)

    # 找到掩码不为零的位置
    mask_indices = np.where(resized_mask != 0)

    # 叠加掩码部分
    composite_image[mask_indices] = masked_np_image[mask_indices]

    # 将合成后的图像转换为 PIL Image 并保存或显示
    composite_image = Image.fromarray(composite_image)
    #composite_image.save("composite_image.png")  # 保存图像
    #composite_image.show()  # 显示图像

    '''
    # compose masked image with generated image
    np_image = np_image + np.array(generated_images[0])
    # save image
    np_image = np_image.astype(np.uint8)
    Image.fromarray(np_image).save(f"hug_lab_masked.png")
    '''

    if refiner:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, cache_dir='/root/autodl-tmp'
        )
        pipe = pipe.to("cuda")
        prompt = "refine the poster"
        generated_images = pipe(prompt, image=composite_image).images
        generated_image = generated_images[0]
        #generated_image.save(f"hug_lab.png")
        return generated_image
    else:
        return composite_image