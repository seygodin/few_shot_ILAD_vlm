#CUDA_VISIBLE_DEVICES=3 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --save True --save_path /home/seungeon/Workspace/vlm/ad_clip/images/capsule --data_name capsule

# CUDA_VISIBLE_DEVICES=3 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --data_name pushpins

CUDA_VISIBLE_DEVICES=3 python make_detection.py --model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k --kernel_size 200 --stride 50 --overlap_count 12 --data_name pcb4 --gpu_number 3