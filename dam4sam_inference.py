import logging
import os
from PIL import Image
from dam4sam_tracker import DAM4SAMTracker
import numpy as np
import torch
from typing import Tuple
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)


def process_masks(masks: torch.Tensor, size: Tuple[int, int]) -> torch.BoolTensor:
    """Resize and process mask tensors into boolean masks.

    Args:
        masks (torch.Tensor): Input masks of shape (N, 1, H, W) or (N, H, W).
        size (Tuple[int, int]): Target size (H, W) for resizing.

    Returns:
        torch.BoolTensor: Boolean masks of shape (N, H, W), where mask regions are True.
    """
    if masks.shape[-2:] != size:
        # Ensure we have 4D tensor for interpolation (N, C, H, W)
        if masks.dim() == 3:  # (N, H, W) -> (N, 1, H, W)
            masks = masks.unsqueeze(1)

        masks = torch.nn.functional.interpolate(masks.float(), size=size, mode="bilinear", align_corners=False)

    if masks.dim() > 3:  # Handle (N, 1, H, W) -> (N, H, W)
        masks = masks.squeeze(1)

    return masks > 0


def compute_iou(
    pred_masks: torch.BoolTensor,
    true_masks: torch.BoolTensor,
    both_absent_as_one: bool = False,
) -> torch.Tensor:
    """Compute Intersection-over-Union (IoU) for binary segmentation masks.

    Args:
        pred_masks (torch.BoolTensor): Predicted masks of shape (B, H, W) or (H, W).
        true_masks (torch.BoolTensor): Ground-truth masks of shape (B, H, W) or (H, W).
        both_absent_as_one (bool): If True, IoU=1 when both masks are entirely empty.

    Returns:
        torch.Tensor: IoU scores of shape (B,).
    """
    if pred_masks.dim() == 2:  # handle single mask
        pred_masks = pred_masks.unsqueeze(0)
        true_masks = true_masks.unsqueeze(0)

    # Convert to uint8 to save memory and speed up bitwise ops
    pred_masks = pred_masks.to(torch.uint8)
    true_masks = true_masks.to(torch.uint8)

    intersection = (pred_masks & true_masks).sum(dim=(1, 2))
    union = (pred_masks | true_masks).sum(dim=(1, 2))

    both_absent = union == 0
    iou = intersection.to(torch.float32) / union.clamp(min=1).to(torch.float32)  # avoid div by 0
    if both_absent_as_one:
        iou[both_absent] = 1.0

    return iou


def tracking_quality(pred_masks: torch.BoolTensor, true_masks: torch.BoolTensor) -> torch.Tensor:
    """Tracking quality metric (IoU, treating both-absent as IoU=1)."""
    return compute_iou(pred_masks, true_masks, both_absent_as_one=True)


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def load_masks_from_dir(input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(input_mask_dir, video_name, object_name, f"{frame_name}.png")
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette


def dam4sam_score(selected_eval_file, eval_resolution, skip_frames=7):
    data_path = "data/datasets/MOSEv2/train"

    with open(selected_eval_file, "r") as f:
        video_names = [line.strip() for line in f if line.strip()]

    total_steps = 0
    scores = []
    for video_name in video_names:
        frame_names = sorted(
            [
                os.path.splitext(p)[0]
                for p in os.listdir(f"{data_path}/JPEGImages/{video_name}")
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
        )

        per_obj_input_mask, _ = load_masks_from_dir(
            input_mask_dir=f"{data_path}/Annotations",
            video_name=video_name,
            frame_name=frame_names[0],
            per_obj_png_file=False,
        )
        input_imgs = [Image.open(f"{data_path}/JPEGImages/{video_name}/{frame_name}.jpg") for frame_name in frame_names]

        for obj_id in per_obj_input_mask.keys():
            mask = per_obj_input_mask[obj_id]

            predictor = DAM4SAMTracker("sam21pp-L", input_image_size=eval_resolution)
            predictor.initialize(input_imgs[0], mask)

            object_score = 0
            object_steps = 0
            skipped_frames = 0

            for frame_idx, input_img in enumerate(tqdm(input_imgs[1:], desc="propagate in video")):
                pred_mask = predictor.track(input_img)["pred_mask"]
                if skipped_frames < skip_frames:
                    skipped_frames += 1
                    continue

                pred_mask = process_masks(torch.tensor(pred_mask).unsqueeze(0), eval_resolution)
                true_mask, _ = load_masks_from_dir(
                    input_mask_dir=f"{data_path}/Annotations",
                    video_name=video_name,
                    frame_name=frame_names[frame_idx],
                    per_obj_png_file=False,
                )
                if obj_id in true_mask:
                    true_mask = process_masks(
                        torch.tensor(true_mask[obj_id], dtype=torch.bool, device=pred_mask.device).unsqueeze(0),
                        size=eval_resolution,
                    )
                else:
                    true_mask = torch.zeros(pred_mask.shape, dtype=torch.bool, device=pred_mask.device)

                quality = tracking_quality(pred_mask, true_mask)
                object_steps += 1
                object_score += quality.item()

            print(object_score / object_steps)
            scores.append(object_score / object_steps)
            total_steps += object_steps
            if total_steps % 1000 == 0:
                print(f"Processed {total_steps} frames...")

    return np.mean(scores), total_steps


def main():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    selected_eval_file = "sam2rl/configs/datasets/valid.txt"
    scores, steps = dam4sam_score(selected_eval_file, eval_resolution=256)
    print(f"DAM4SAM: quality: {scores}, steps: {steps}")


if __name__ == "__main__":
    main()
