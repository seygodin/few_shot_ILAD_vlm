#!/bin/sh

python make_detection.py --gpu_number 2 --data_name blue_splicing --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --save True --save_path /home/seungeon/Workspace/vlm/ad_clip/images/blue_splicing/
python make_detection.py --gpu_number 2 --data_name red_splicing --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --save True --save_path /home/seungeon/Workspace/vlm/ad_clip/images/red_splicing/
python make_detection.py --gpu_number 2 --data_name yellow_splicing --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --save True --save_path /home/seungeon/Workspace/vlm/ad_clip/images/yellow_splicing/