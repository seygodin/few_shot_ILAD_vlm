#!/bin/sh

a=0
while [ $a -lt $1 ]
do
    CUDA_VISIBLE_DEVICES=$2 python train_wrong_pos_rule.py --detection True --preprocess True --few_shot True --lr $8 --epoch 100 --gpu_number $2 --data_name $3 --mask $4 --text $5 --shot $6 --double_encoder $7 --neg_loss $9 --ris 0 --num_negative $10
    #CUDA_VISIBLE_DEVICES=$2 python train_review.py --detection True --preprocess True --few_shot True --lr $8 --epoch 100 --gpu_number $2 --data_name $3 --mask $4 --text $5 --shot $6 --double_encoder $7 --neg_loss $9 --ris 0 --num_negative $10
    #CUDA_VISIBLE_DEVICES=$2 python train_glclip.py --detection True --preprocess True --few_shot True --lr $8 --epoch 60 --gpu_number $2 --data_name $3 --mask $4 --text $5 --shot $6 --double_encoder $7 --neg_loss $9 --ris 0 --num_negative $10
    a=`expr $a + 1`
done