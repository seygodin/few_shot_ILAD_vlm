from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


model_dict = {"vit_h": "/home/seungeon/Workspace/vlm/sam_model_checkpoints/sam_vit_h_4b8939.pth",
              "vit_l": "/home/seungeon/Workspace/vlm/sam_model_checkpoints/sam_vit_l_0b3195.pth",
              "vit_b": "/home/seungeon/Workspace/vlm/sam_model_checkpoints/sam_vit_b_01ec64.pth"}

#sam = sam_model_registry["vit_h"](checkpoint="/home/seungeon/Workspace/vlm/sam_model_checkpoints/sam_vit_h_4b8939.pth")


def get_sam_model(model_name: str, device:str="cpu"):
    if model_name not in model_dict.keys():
        raise NotImplementedError(f"The supported models are {model_dict.keys()}")
    else:
        sam = sam_model_registry[model_name](checkpoint=model_dict[model_name])

        sam.to(device=device)

        sam_mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
        return sam_mask_generator