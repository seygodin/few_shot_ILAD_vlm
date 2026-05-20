CUDA_VISIBLE_DEVICES=0 python global_local_clip.py --data_name breakfast
#CUDA_VISIBLE_DEVICES=0 python global_local_clip.py --data_name banana_juice

#CUDA_VISIBLE_DEVICES=0 python make_detection.py --data_name breakfast --save True --ris 1 --kernel_size 250 --stride 50 --overlap_count 3 --save_path /home/seungeon/Workspace/vlm/ad_clip/ris_images/breakfast/ --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k
#CUDA_VISIBLE_DEVICES=0 python make_detection.py --data_name banana_juice --save True --ris 1 --kernel_size 400 --stride 50 --overlap_count 2 --save_path /home/seungeon/Workspace/vlm/ad_clip/ris_images/banana_juice/ --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k
