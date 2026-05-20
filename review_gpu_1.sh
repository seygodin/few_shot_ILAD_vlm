a=0
b=10
while [ $a -lt $b ]
do
    #CUDA_VISIBLE_DEVICES=$2 python train.py --detection True --preprocess True --few_shot True --lr $8 --epoch 60 --gpu_number $2 --data_name $3 --mask $4 --text $5 --shot $6 --double_encoder $7 --neg_loss $9 --ris 0 --num_negative $10
    CUDA_VISIBLE_DEVICES=1 python train_review.py --double_encoder True --detection True --preprocess True --few_shot True --lr 1e-4 --epoch 60 --log True --data_name breakfast --mask True --text True --ris 0 --kernel_size $1 --overlap_count $2 --shot $3
    a=`expr $a + 1`
done