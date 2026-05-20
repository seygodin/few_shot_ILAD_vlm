# CUDA_VISIBLE_DEVICES=0 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --save True --save_path /home/seungeon/Workspace/vlm/ad_clip/images/transistor --data_name transistor

# CUDA_VISIBLE_DEVICES=0 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --save True --save_path /home/seungeon/Workspace/vlm/ad_clip/images/pcb1 --data_name pcb1
# CUDA_VISIBLE_DEVICES=0 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --save True --save_path /home/seungeon/Workspace/vlm/ad_clip/images/pcb2 --data_name pcb2

#!/bin/sh

# CUDA_VISIBLE_DEVICES=0 python make_detection.py --data_name blue_splicing --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 250 --stride 50 --overlap_count 8
# CUDA_VISIBLE_DEVICES=0 python make_detection.py --data_name red_splicing --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 250 --stride 50 --overlap_count 8
# CUDA_VISIBLE_DEVICES=0 python make_detection.py --data_name yellow_splicing --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 250 --stride 50 --overlap_count 8


CUDA_VISIBLE_DEVICES=0 python make_detection.py --data_name transistor --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 250 --stride 50 --overlap_count 8
CUDA_VISIBLE_DEVICES=0 python make_detection.py --data_name pcb1 --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 250 --stride 50 --overlap_count 8
