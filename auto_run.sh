#!/bin/sh

product_name="$1"
gpu_number="$2"
dir_nam="$3" 
lr="$4"

# 제품 이름을 사용하여 다양한 조건으로 스크립트를 실행합니다.
./run.sh 10 "$gpu_number" "$product_name" False False 5 False "$lr" 10 1 > "${dir_nam}/${product_name}_5_f_f_result"
./run.sh 10 "$gpu_number" "$product_name" True False 5 True "$lr" 10 1 > "${dir_nam}/${product_name}_5_t_f_result"
./run.sh 10 "$gpu_number" "$product_name" False True 5 False "$lr" 10 1 > "${dir_nam}/${product_name}_5_f_t_result"
./run.sh 10 "$gpu_number" "$product_name" True True 5 True "$lr" 10 1 > "${dir_nam}/${product_name}_5_t_t_result"
./run.sh 10 "$gpu_number" "$product_name" False False 1 False "$lr" 10 1 > "${dir_nam}/${product_name}_1_f_f_result"
./run.sh 10 "$gpu_number" "$product_name" True False 1 True "$lr" 10 1 > "${dir_nam}/${product_name}_1_t_f_result"
./run.sh 10 "$gpu_number" "$product_name" False True 1 False "$lr" 10 1 > "${dir_nam}/${product_name}_1_f_t_result"
./run.sh 10 "$gpu_number" "$product_name" True True 1 True "$lr" 10 1 > "${dir_nam}/${product_name}_1_t_t_result"