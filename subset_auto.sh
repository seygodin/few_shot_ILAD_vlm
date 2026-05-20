#!/bin/sh

gpu_number="$1"
product_name1="$2"
product_name2="$3"
product_name3="$4"

./auto.sh "$product_name1" "$gpu_number"
./auto.sh "$product_name2" "$gpu_number"
./auto.sh "$product_name3" "$gpu_number"