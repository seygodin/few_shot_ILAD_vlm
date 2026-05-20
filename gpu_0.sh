#!/bin/sh

./auto.sh breakfast 0 accuracy 1e-4 4
./auto.sh banana_juice 0 accuracy 1e-4 4

#./auto.sh breakfast 0 num_pos_3 1e-4 3
#./auto.sh banana_juice 0 num_pos_3 1e-4 3

#./review_gpu_0.sh 200 8 1 > review_record/shot_1_k_200_t_8.txt
#./review_gpu_0.sh 200 8 5 > review_record/shot_5_k_200_t_8.txt
#./review_gpu_0.sh 200 10 1 > review_record/shot_1_k_200_t_10.txt

#이셋은 이따가 1, 2, 3번 끝나고 돌리기
#./review_gpu_0.sh 200 10 5 > review_record/shot_5_k_200_t_10.txt

#./review_gpu_0.sh 200 12 1 > review_record/shot_1_k_200_t_12.txt
#./review_gpu_0.sh 200 12 5 > review_record/shot_5_k_200_t_12.txt
#./auto.sh pcb1 0 additional_data 1e-4 5
#ㄴ./auto.sh pcb2 0 additional_data 1e-4 5