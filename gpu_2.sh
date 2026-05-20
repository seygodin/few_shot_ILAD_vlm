#!/bin/sh

./auto.sh screw_bag 2 accuracy 1e-4 4
./auto.sh blue_splicing 2 accuracy 1e-4 4

#./review_gpu_2.sh 300 8 1 > review_record/shot_1_k_300_t_8.txt
#./review_gpu_2.sh 300 8 5 > review_record/shot_5_k_300_t_8.txt

#./review_gpu_2.sh 300 10 1 > review_record/shot_1_k_300_t_10.txt
#./review_gpu_2.sh 300 10 5 > review_record/shot_5_k_300_t_10.txt
#./auto.sh cable 2 additional_data 1e-4 5