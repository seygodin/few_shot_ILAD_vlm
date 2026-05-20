# CUDA_VISIBLE_DEVICES=1 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --save True --save_path /home/seungeon/Workspace/vlm/ad_clip/images/transistor --data_name transistor

# CUDA_VISIBLE_DEVICES=1 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --save True --save_path /home/seungeon/Workspace/vlm/ad_clip/images/pcb1 --data_name pcb1
# CUDA_VISIBLE_DEVICES=1 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --save True --save_path /home/seungeon/Workspace/vlm/ad_clip/images/pcb2 --data_name pcb2
# CUDA_VISIBLE_DEVICES=1 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --data_name banana_juice
# CUDA_VISIBLE_DEVICES=1 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --data_name orange_juice
# CUDA_VISIBLE_DEVICES=1 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --data_name cherry_juice

CUDA_VISIBLE_DEVICES=1 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --data_name capsule --gpu_number 1
CUDA_VISIBLE_DEVICES=1 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --data_name pcb2 --gpu_number 1