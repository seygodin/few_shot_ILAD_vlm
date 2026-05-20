CUDA_VISIBLE_DEVICES=1 python global_local_clip.py --data_name pushpins
CUDA_VISIBLE_DEVICES=1 python global_local_clip.py --data_name orange_juice

#CUDA_VISIBLE_DEVICES=1 python make_detection.py --data_name pushpins --save True --ris 1 --kernel_size 400 --stride 50 --overlap_count 2 --save_path /home/seungeon/Workspace/vlm/ad_clip/ris_images/pushpins/ --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k
#CUDA_VISIBLE_DEVICES=1 python make_detection.py --data_name orange_juice --save True --ris 1 --kernel_size 400 --stride 50 --overlap_count 2 --save_path /home/seungeon/Workspace/vlm/ad_clip/ris_images/orange_juice/ --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k
