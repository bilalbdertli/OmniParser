from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image
import io


import base64, os
from utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import torch
from PIL import Image

yolo_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
# caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2")

import re 

import re

def color_to_rgb(color_str: str):
    color_str = color_str.strip().lower()

    # Check for hex: #rrggbb
    if color_str.startswith('#'):
        hex_color = color_str.lstrip('#')
        if len(hex_color) != 6:
            raise ValueError(f"Expected a 6-digit hex code, got: '{hex_color}'")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Check for rgb(...) or rgba(...)
    if color_str.startswith('rgb'):
        # Extract numeric parts
        numbers = re.findall(r'[\d.]+', color_str)
        if len(numbers) < 3:
            raise ValueError(f"Invalid rgb/rgba format: '{color_str}'")
        r, g, b = map(float, numbers[:3])
        return (int(r), int(g), int(b))

    # Check for hsl(...) format
    if color_str.startswith('hsl'):
        # Extract H, S, L (and ignore % signs)
        numbers = re.findall(r'[\d.]+', color_str)
        if len(numbers) < 3:
            raise ValueError(f"Invalid hsl format: '{color_str}'")
        h, s, l = map(float, numbers[:3])
        # Convert S and L from percentage to fraction
        s /= 100.0
        l /= 100.0

        # Convert HSL to RGB
        r, g, b = hsl_to_rgb(h, s, l)
        return (int(r), int(g), int(b))

    raise ValueError(f"Unrecognized color format: '{color_str}'")


def hsl_to_rgb(h, s, l):
    # H is in degrees: convert to fraction of 360
    h = (h % 360) / 360.0

    def hue_to_rgb(p, q, t):
        if t < 0: 
            t += 1
        if t > 1: 
            t -= 1
        if t < 1/6:
            return p + (q - p) * 6 * t
        if t < 1/2:
            return q
        if t < 2/3:
            return p + (q - p) * (2/3 - t) * 6
        return p

    if s == 0:
        # Achromatic (gray)
        r = g = b = l * 255
        return r, g, b

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    r = 255 * hue_to_rgb(p, q, h + 1/3)
    g = 255 * hue_to_rgb(p, q, h)
    b = 255 * hue_to_rgb(p, q, h - 1/3)
    return r, g, b

MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

DEVICE = torch.device('cuda')

# @spaces.GPU
# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process(
    image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz,
    bounding_box_hex,
    text_hex
) -> Optional[Image.Image]:
    # Convert hex to RGB
    bounding_box_rgb = color_to_rgb(bounding_box_hex)
    text_rgb = color_to_rgb(text_hex)
    image_save_path = 'imgs/saved_image_demo.png'
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 4 * box_overlay_ratio,  # Increased text scale
        'text_thickness': max(int(4 * box_overlay_ratio), 2),  # Increased text thickness
        'text_padding': max(int(15 * box_overlay_ratio), 5),  # Increased text padding
        'thickness': max(int(15 * box_overlay_ratio), 5),  # Increased border thickness
        'bounding_box_color': bounding_box_rgb,
        'text_color_rgb': text_rgb
        
    }
    # import pdb; pdb.set_trace()

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_save_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    # print('prompt:', prompt)
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_save_path, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz)  
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print('finish processing')
    parsed_content_list = '\n'.join(parsed_content_list)
    return image, str(parsed_content_list)



with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(
                type='pil', label='Upload image')
            # set the threshold for removing the bounding boxes with low confidence, default is 0.05
            box_threshold_component = gr.Slider(
                label='Box Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.05)
            # set the threshold for removing the bounding boxes with large overlap, default is 0.1
            iou_threshold_component = gr.Slider(
                label='IOU Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.1)
            use_paddleocr_component = gr.Checkbox(
                label='Use PaddleOCR', value=True)
            imgsz_component = gr.Slider(
                label='Icon Detect Image Size', minimum=640, maximum=1920, step=32, value=640)
            with gr.Row():
                bounding_box_color_component = gr.ColorPicker(
                    label='Bounding Box Color', value='#000000'
                )
                text_color_component = gr.ColorPicker(
                    label='Text Color', value='#FFFFFF'
                )
            submit_button_component = gr.Button(
                value='Submit', variant='primary')
        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image Output')
            text_output_component = gr.Textbox(label='Parsed screen elements', placeholder='Text Output')

    submit_button_component.click(
        fn=process,
        inputs=[
            image_input_component,
            box_threshold_component,
            iou_threshold_component,
            use_paddleocr_component,
            imgsz_component,
            bounding_box_color_component,
            text_color_component
        ],
        outputs=[image_output_component, text_output_component]
    )

# demo.launch(debug=False, show_error=True, share=True)
demo.launch(share=True, server_port=7861, server_name='0.0.0.0')