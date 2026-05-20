#!/bin/sh

./auto.sh red_splicing 3 accuracy 1e-4 4
./auto.sh yellow_splicing 3 accuracy 1e-4 4
./auto.sh orange_juice 3 accuracy 1e-4 4
./auto.sh cherry_juice 3 accuracy 1e-4 4

#./review_gpu_3.sh 300 12 1 > review_record/shot_1_k_300_t_12.txt
#./review_gpu_3.sh 300 12 5 > review_record/shot_5_k_300_t_12.txt

#./review_gpu_3.sh 250 12 1 > review_record/shot_1_k_250_t_12.txt
#./review_gpu_3.sh 250 12 5 > review_record/shot_5_k_250_t_12.txt

#./auto.sh capsule 3 additional_data 1e-4 5
#./auto.sh transistor 3 additional_data 1e-4 5
