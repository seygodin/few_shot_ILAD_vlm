#!/bin/sh

./auto.sh pushpins 1 accuracy 1e-4 4
./auto.sh cherry_juice 1 accuracy 1e-4 4


#./review_gpu_1.sh 250 8 1 > review_record/shot_1_k_250_t_8.txt
#./review_gpu_1.sh 250 8 5 > review_record/shot_5_k_250_t_8.txt

#./review_gpu_1.sh 250 10 1 > review_record/shot_1_k_250_t_10.txt
#./review_gpu_1.sh 250 10 5 > review_record/shot_5_k_250_t_10.txt
#./auto.sh pcb3 1 additional_data 1e-4 5
#./auto.sh pcb4 1 additional_data 1e-4 5