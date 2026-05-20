#CUDA_VISIBLE_DEVICES=3 python global_local_clip.py --data_name red_splicing
#CUDA_VISIBLE_DEVICES=3 python global_local_clip.py --data_name blue_splicing
CUDA_VISIBLE_DEVICES=3 python global_local_clip.py --data_name yellow_splicing

#CUDA_VISIBLE_DEVICES=2 python make_detection.py --data_name blue_splicing --save True --ris 1 --kernel_size 400 --stride 50 --overlap_count 2 --save_path /home/seungeon/Workspace/vlm/ad_clip/ris_images/blue_splicing/ --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k
#CUDA_VISIBLE_DEVICES=2 ython make_detection.py --data_name red_splicing --save True --ris 1 --kernel_size 400 --stride 50 --overlap_count 2 --save_path /home/seungeon/Workspace/vlm/ad_clip/ris_images/red_splicing/ --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k
#CUDA_VISIBLE_DEVICES=2 python make_detection.py --data_name yellow_splicing --save True --ris 1 --kernel_size 400 --stride 50 --overlap_count 2 --save_path /home/seungeon/Workspace/vlm/ad_clip/ris_images/yellow_splicing/ --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k

#CUDA_VISIBLE_DEVICES=2 python make_detection.py --data_name cherry_juice --save True --ris 1 --kernel_size 400 --stride 50 --overlap_count 2 --save_path /home/seungeon/Workspace/vlm/ad_clip/ris_images/cherry_juice/ --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k
