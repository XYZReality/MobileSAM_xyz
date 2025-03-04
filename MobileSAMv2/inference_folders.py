import argparse
import torch
from PIL import Image
import cv2
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import sam_model_registry, SamPredictor
from typing import Any, Dict, Generator, List
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ObjectAwareModel_path", type=str, default='./PromptGuidedDecoder/ObjectAwareModel.pt', help="ObjectAwareModel path")
    parser.add_argument("--Prompt_guided_Mask_Decoder_path", type=str, default='./PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt', help="Prompt_guided_Mask_Decoder path")
    parser.add_argument("--encoder_path", type=str, default="./", help="select your own path")
    parser.add_argument("--img_path", type=str, default="./test_images/", help="path to image file")
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument("--iou", type=float, default=0.9, help="yolo iou")
    parser.add_argument("--conf", type=float, default=0.4, help="yolo object confidence threshold")
    parser.add_argument("--retina", type=bool, default=True, help="draw segmentation masks")
    parser.add_argument("--output_dir", type=str, default="./output/", help="output directory")
    parser.add_argument("--encoder_type", choices=['tiny_vit','sam_vit_h','mobile_sam','efficientvit_l2','efficientvit_l1','efficientvit_l0'], help="choose the model type")
    parser.add_argument("--no-json", action="store_true", help="Disable saving of mask details to JSON files")
    parser.add_argument("--include-pixels", action="store_true", help="Include all pixel coordinates in JSON (can make files very large)")
    return parser.parse_args()

def create_model():
    Prompt_guided_path='./PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt'
    obj_model_path='./weight/ObjectAwareModel.pt'
    ObjAwareModel = ObjectAwareModel(obj_model_path)
    PromptGuidedDecoder=sam_model_registry['PromptGuidedDecoder'](Prompt_guided_path)
    mobilesamv2 = sam_model_registry['vit_h']()
    mobilesamv2.prompt_encoder=PromptGuidedDecoder['PromtEncoder']
    mobilesamv2.mask_decoder=PromptGuidedDecoder['MaskDecoder']
    return mobilesamv2, ObjAwareModel
    
def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

def get_color_map_list(num_classes=256):
    """
    Returns a color map list with length equal to num_classes that can be used
    to visualize segmentation masks

    Args:
        num_classes (int): Number of colors to generate
        
    Returns:
        List[int]: List of RGB colors
    """
    color_map = []
    for i in range(num_classes):
        r = (i * (1 + i % 10)) % 255
        g = (i * (5 + i % 15)) % 255
        b = (i * (10 + i % 20)) % 255
        color_map.extend([r, g, b])
    return color_map

def visualize(image, result, color_map, weight=0.6):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (np.ndarray): The input image array.
        result (np.ndarray): The predict result of image.
        color_map (list): The color used to save the prediction results.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6

    Returns:
        vis_result (np.ndarray): The visualized result.
    """
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c3, c2, c1))

    vis_result = cv2.addWeighted(image, weight, pseudo_img, 1 - weight, 0)
    return vis_result

encoder_path = {
    'efficientvit_l2': './weight/l2.pt',
    'tiny_vit': './weight/mobile_sam.pt',
    'sam_vit_h': './weight/sam_vit_h.pt',
}

def main(args):
    # Create output directories
    output_folder = Path(args.output_dir)
    output_mask_dir = output_folder / "masks"
    output_overlay_dir = output_folder / "overlays"
    
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    output_overlay_dir.mkdir(parents=True, exist_ok=True)
    
    # Create JSON output directory by default unless --no-json is specified
    output_json_dir = None
    if not args.no_json:
        output_json_dir = output_folder / "json"
        output_json_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    mobilesamv2, ObjAwareModel = create_model()
    image_encoder = sam_model_registry[args.encoder_type](encoder_path[args.encoder_type])
    mobilesamv2.image_encoder = image_encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobilesamv2.to(device=device)
    mobilesamv2.eval()
    predictor = SamPredictor(mobilesamv2)
    
    # Get list of image files in input directory
    input_folder = Path(args.img_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(input_folder.glob(f'*{ext}')))
        image_files.extend(list(input_folder.glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"No image files found in {args.img_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Track total processing time
    total_start_time = time.time()
    
    # Get color map list once for consistent colors
    color_map_list = get_color_map_list(256)
    
    # Process each image
    for img_path in image_files:
        start_time = time.time()
        image_name = img_path.name
        print(f"Processing {image_name}...")
        
        # Read the image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        # Convert to RGB for model input
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get object detection results
        obj_results = ObjAwareModel(image_rgb, device=device, retina_masks=args.retina, 
                                   imgsz=args.imgsz, conf=args.conf, iou=args.iou)
        
        # Set image for prediction
        predictor.set_image(image_rgb)
        
        # Get bounding boxes
        input_boxes1 = obj_results[0].boxes.xyxy
        input_boxes = input_boxes1.cpu().numpy()
        input_boxes = predictor.transform.apply_boxes(input_boxes, predictor.original_size)
        input_boxes = torch.from_numpy(input_boxes).to(device)
        
        # Process masks in batches
        sam_mask = []
        image_embedding = predictor.features
        image_embedding = torch.repeat_interleave(image_embedding, 320, dim=0)
        prompt_embedding = mobilesamv2.prompt_encoder.get_dense_pe()
        prompt_embedding = torch.repeat_interleave(prompt_embedding, 320, dim=0)
        
        for (boxes,) in batch_iterator(320, input_boxes):
            with torch.no_grad():
                img_embed = image_embedding[0:boxes.shape[0],:,:,:]
                prompt_embed = prompt_embedding[0:boxes.shape[0],:,:,:]
                sparse_embeddings, dense_embeddings = mobilesamv2.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None,)
                low_res_masks, _ = mobilesamv2.mask_decoder(
                    image_embeddings=img_embed,
                    image_pe=prompt_embed,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    simple_type=True,
                )
                # Post-process masks to original image size
                low_res_masks = predictor.model.postprocess_masks(
                    low_res_masks, predictor.input_size, predictor.original_size
                )
                sam_mask_pre = (low_res_masks > mobilesamv2.mask_threshold) * 1.0
                sam_mask.append(sam_mask_pre.squeeze(1))
        
        # Concatenate all masks
        if len(sam_mask) > 0:
            sam_mask = torch.cat(sam_mask)
            
            # Sort masks by area (largest first)
            areas = torch.sum(sam_mask, dim=(1, 2))
            sorted_indices = torch.argsort(areas, descending=True)
            sorted_masks = sam_mask[sorted_indices]
            
            # Create an indexed mask for visualization
            h, w = sorted_masks.shape[1:3]
            indexed_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Assign each mask a unique ID
            for i, mask in enumerate(sorted_masks):
                # Add 1 to avoid 0 (background)
                mask_id = i + 1
                if mask_id >= 255:  # Avoid overflow
                    break
                indexed_mask[mask.cpu().numpy() > 0] = mask_id
            
            # Create colorized mask and overlay
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(1, 256):
                r = color_map_list[i*3]
                g = color_map_list[i*3+1]
                b = color_map_list[i*3+2]
                colored_mask[indexed_mask == i] = [b, g, r]  # BGR for OpenCV
            
            # Create overlay
            overlay_img = visualize(image, indexed_mask, color_map=color_map_list)
            
            # Get output filenames
            name, ext = os.path.splitext(image_name)
            output_mask_path = output_mask_dir / f"{name}_mask{ext}"
            output_overlay_path = output_overlay_dir / f"{name}_overlay{ext}"
            
            # Save the images
            cv2.imwrite(str(output_mask_path), colored_mask)
            cv2.imwrite(str(output_overlay_path), overlay_img)
            
            image_mask_dir = output_mask_dir / name
            image_mask_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each mask separately
            for i, mask_tensor in enumerate(sorted_masks):
                if i >= 255:  # Avoid overflow
                    break
                
                # Convert mask to binary numpy array (255 for white, 0 for black)
                binary_mask = (mask_tensor.cpu().numpy() > 0).astype(np.uint8) * 255
                
                # Save individual mask
                mask_filename = f"mask_{i+1}{ext}"
                cv2.imwrite(str(image_mask_dir / mask_filename), binary_mask)
        
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Save mask details to JSON if requested
            if output_json_dir:
                output_json_path = output_json_dir / f"{name}_masks.json"
                
                # Extract relevant data from masks
                mask_data = []
                for i, mask_tensor in enumerate(sorted_masks):
                    if i >= 255:  # Avoid overflow
                        break
                        
                    # Convert mask to numpy for processing
                    mask = mask_tensor.cpu().numpy()
                    mask_id = i + 1  # Match the ID used in indexed_mask
                    
                    # Get mask color
                    r = color_map_list[mask_id*3]
                    g = color_map_list[mask_id*3+1]
                    b = color_map_list[mask_id*3+2]
                    
                    # Find mask bounding box
                    if np.sum(mask) > 0:  # Only process non-empty masks
                        y_indices, x_indices = np.where(mask > 0)
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        y_min, y_max = np.min(y_indices), np.max(y_indices)
                        
                        # Prepare pixel coordinates (if requested)
                        pixel_coords = []
                        if args.include_pixels:
                            for y, x in zip(y_indices, x_indices):
                                pixel_coords.append([int(y), int(x)])
                        
                        # Get score from obj_results if available
                        score = None
                        if i < len(sorted_indices) and i < len(obj_results[0].boxes.conf):
                            original_idx = sorted_indices[i].item()
                            if original_idx < len(obj_results[0].boxes.conf):
                                score = float(obj_results[0].boxes.conf[original_idx].cpu().numpy())
                        
                        mask_info = {
                            "id": mask_id,
                            "color": [int(r), int(g), int(b)],
                            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                            "area": int(np.sum(mask)),
                            "score": score,
                            "pixel_count": len(y_indices),
                        }
                        
                        # Include all pixel coordinates if requested
                        if args.include_pixels:
                            mask_info["pixels"] = pixel_coords
                        else:
                            # Include a few sample points
                            sample_size = min(10, len(y_indices))
                            sample_indices = np.random.choice(len(y_indices), sample_size, replace=False)
                            sample_points = []
                            for idx in sample_indices:
                                sample_points.append([int(y_indices[idx]), int(x_indices[idx])])
                            mask_info["sample_points"] = sample_points
                        
                        mask_data.append(mask_info)
                
                # Save to JSON file
                with open(output_json_path, 'w') as f:
                    json.dump({
                        "image": image_name,
                        "image_size": [image.shape[1], image.shape[0]],  # [width, height]
                        "mask_count": len(mask_data),
                        "processing_time_seconds": round(processing_time, 3),
                        "masks": mask_data
                    }, f, indent=2)
                
                print(f"Processed {image_name} in {processing_time:.2f}s - saved to {output_mask_path}, {output_overlay_path}, and {output_json_path}")
            else:
                print(f"Processed {image_name} in {processing_time:.2f}s - saved to {output_mask_path} and {output_overlay_path}")
        else:
            print(f"No masks generated for: {image_name}")
    
    total_time = time.time() - total_start_time
    print(f"Processing complete. Processed {len(image_files)} images in {total_time:.2f} seconds.")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)