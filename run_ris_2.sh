CUDA_VISIBLE_DEVICES=2 python global_local_clip.py --data_name screw_bag
CUDA_VISIBLE_DEVICES=2 python global_local_clip.py --data_name cherry_juice

#CUDA_VISIBLE_DEVICES=2 python make_detection.py --data_name screw_bag --save True --ris 1 --kernel_size 400 --stride 50 --overlap_count 2 --save_path /home/seungeon/Workspace/vlm/ad_clip/ris_images/screw_bag/ --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k