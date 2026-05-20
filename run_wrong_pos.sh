#!/bin/sh

a=0
while [ $a -lt $1 ]
do
    #CUDA_VISIBLE_DEVICES=$2 python train_wrong_pos_rule.py --detection True --preprocess True --few_shot True --lr $8 --epoch 50 --gpu_number $2 --data_name $3 --mask $4 --text $5 --shot $6 --double_encoder $7 --neg_loss $9 --ris 0 --num_rule $10
    CUDA_VISIBLE_DEVICES=$2 python train.py --detection True --preprocess True --few_shot True --lr 1e-4 --epoch 50 --gpu_number 0 --data_name breakfast --mask False --text False --shot 1 --double_encoder True --ris 0
    #CUDA_VISIBLE_DEVICES=$2 python train_review.py --detection True --preprocess True --few_shot True --lr $8 --epoch 100 --gpu_number $2 --data_name $3 --mask $4 --text $5 --shot $6 --double_encoder $7 --neg_loss $9 --ris 0 --num_negative $10
    #CUDA_VISIBLE_DEVICES=$2 python train_glclip.py --detection True --preprocess True --few_shot True --lr $8 --epoch 60 --gpu_number $2 --data_name $3 --mask $4 --text $5 --shot $6 --double_encoder $7 --neg_loss $9 --ris 0 --num_negative $10
    a=`expr $a + 1`
done