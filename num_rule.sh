product_name="$1"
gpu_number="$2"

echo "$product_name"
echo 1
CUDA_VISIBLE_DEVICES="$gpu_number" python train.py --detection True --preprocess True --mask False --text False --few_shot True --shot 1 --data_name "$product_name" --lr 1e-4 --num_rule 1 --seed 1234
CUDA_VISIBLE_DEVICES="$gpu_number" python train.py --detection True --preprocess True --mask False --text False --few_shot True --shot 1 --data_name "$product_name" --lr 1e-4 --num_rule 3 --seed 1234
