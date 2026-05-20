import open_clip
from ad_clip.utils_save import get_rule, get_data_path, get_rule_tokens, get_object_info
from ad_clip.model import get_model
from ad_clip.data import ad_dataset, ad_obj_dataset, ad_ris_dataset, ad_aug_dataset
import argparse

import os
import torch

import numpy as np
import random


def set_random_seed(random_seed=123):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # multi-GPU
    print(f"Random seed {random_seed} is selected.")


def main(args):
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained_name)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    print("Tokenizing rules")
    #rules_for_data = get_rule(data_name = args.data_name, rule_idxs=[i for i in range(7)])
    #rule_token_pairs = get_rule_tokens(rules=rules_for_data, tokenizer=tokenizer, device=device)

    print("Generating dataloaders")

    if args.ris != 1:
        object_names, objects_for_rules, _ = get_object_info(args.data_name, ris_info=False)
    elif args.ris == 1:
        object_names, objects_for_rules, _, ris_info = get_object_info(args.data_name, ris_info=True)
        sam_text_list = ris_info[0]
        sam_num_list = ris_info[1]
    

    data_base_path_dict = {
                            "breakfast" : "/data/seungeon/orig/breakfast_box",
                            "juice_bot" : "/data/seungeon/orig/juice_bottle",
                            "pushpins" :"/data/seungeon/orig/pushpins",
                            "screw_bag" : "/data/seungeon/orig/screw_bag",
                            "splicing" : "/data/seungeon/orig/splicing_connectors",
                            "banana_juice" : "/data/seungeon/orig/juice_bottle_banana",
                            "cherry_juice" : "/data/seungeon/orig/juice_bottle_cherry",
                            "orange_juice" : "/data/seungeon/orig/juice_bottle_orange",

                            "blue_splicing" : "/data/seungeon/orig/splicing_connector_blue_cable",
                            "red_splicing" : "/data/seungeon/orig/splicing_connector_red_cable",
                            "yellow_splicing" : "/data/seungeon/orig/splicing_connector_yellow_cable",

                            #additional Visa dataset
                            "pcb1" : "/data3/seungeon/data/image/visa/pcb1",
                            "pcb2" : "/data3/seungeon/data/image/visa/pcb2",
                            "pcb3" : "/data3/seungeon/data/image/visa/pcb3",
                            "pcb4" : "/data3/seungeon/data/image/visa/pcb4",

                            #additional MVTec AD dataset
                            "cable": "/data3/seungeon/data/image/mvtec_ad/cable",
                            "capsule": "/data3/seungeon/data/image/mvtec_ad/capsule",
                            "transistor": "/data3/seungeon/data/image/mvtec_ad/transistor",
                            }
    
    base_save_path = data_base_path_dict[args.data_name]

    data_path_dict = get_data_path(args.data_name)
    
    tag = f"k_{args.kernel_size}_t_{args.overlap_count}"
    
    if not os.path.exists(os.path.join(base_save_path, f'{tag}_train.pt')):
        train_dataset = ad_obj_dataset(data_path_dict, 'train_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                    object_names=object_names, target_objects_list=objects_for_rules, overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride,
                                    save=args.save, save_path=os.path.join(args.save_path, "train/good"))
        torch.save(train_dataset, os.path.join(base_save_path, f'{tag}_train.pt'))
        print("Train image saved")
    else:
        print("Train image already exists")

    if not os.path.exists(os.path.join(base_save_path, f'{tag}_val.pt')):
        val_dataset = ad_obj_dataset(data_path_dict, 'val_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                    object_names=object_names, target_objects_list=objects_for_rules, overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride,
                                    save=args.save, save_path=os.path.join(args.save_path, "validation/good"))
        torch.save(val_dataset, os.path.join(base_save_path, f'{tag}_val.pt'))
        print("Valid image saved")
    else:
        print("Valid image already exists")

    if not os.path.exists(os.path.join(base_save_path, f'{tag}_test_good.pt')):
        test_good_dataset = ad_obj_dataset(data_path_dict, 'test_good_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                    object_names=object_names, target_objects_list=objects_for_rules, overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride,
                                    save=args.save, save_path=os.path.join(args.save_path, "test/good"))
        torch.save(test_good_dataset, os.path.join(base_save_path, f'{tag}_test_good.pt'))
        print("Test good image saved")
    else:
        print("Test good image already exists")
    
    if not os.path.exists(os.path.join(base_save_path, f'{tag}_test_la.pt')):
        test_la_dataset = ad_obj_dataset(data_path_dict, 'test_la_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                    object_names=object_names, target_objects_list=objects_for_rules, overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride,
                                    save=args.save, save_path=os.path.join(args.save_path, "test/logical_anomalies"))
        torch.save(test_la_dataset, os.path.join(base_save_path, f'{tag}_test_la.pt'))
        print("Test la saved")
    else:
        print("Test la image already exists")

   
    print("Program end")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--gpu_number', default="0", type=str)
    parser.add_argument('--data_name', default='breakfast', type=str)
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--tag', default='default', type=str)
    parser.add_argument('--model_name', default='ViT-g-14', type=str)   #ViT-g-14
    parser.add_argument('--pretrained_name', default='laion2b_s12b_b42k', type=str)     #laion2b_s12b_b42k
    parser.add_argument('--lora', default=-1, type=int)
    parser.add_argument('--default', default=False, type=bool)
    parser.add_argument('--foundation', default="CLIP", type=str)
    parser.add_argument('--kernel_size', default=250, type=int)
    parser.add_argument('--stride', default=50, type=int)
    parser.add_argument('--overlap_count', default=7, type=int) 
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--ris', default=0, type=int)
    args = parser.parse_args()

    print(args)

    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    set_random_seed(random_seed=args.seed)

    print(f"Torch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is loaded for process")

    

    main(args)

"""
Hyper Parameter Saving

[SAM RIS with CLIP seg]
data_name   kernel_size     overlap_count   stride
breakfast   250                 3           50
juice_bot   400                 2           50
pushpins    400                 2           50
screw_bag   400                 2           50
splicing    400                 2           50
clip_model_name     ViT-g-14
pretrained_name     laion2b_s12b_b42k
sam_model_name      sam_hq_vit_l
sam_model_type      vit_l
"""