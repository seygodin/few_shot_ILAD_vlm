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
                            "breakfast" : "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/breakfast_box",
                            "juice_bot" : "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/juice_bottle",
                            "pushpins" :"/data3/seungeon/data/image/mvtec_loco_anomaly_detection/pushpins",
                            "screw_bag" : "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/screw_bag",
                            "splicing" : "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/splicing_connectors",
                            "banana_juice" : "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/banana_juice",
                            "cherry_juice" : "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/cherry_juice",
                            "orange_juice" : "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/orange_juice",

                            "blue_splicing" : "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/blue_splicing",
                            "red_splicing" : "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/red_splicing",
                            "yellow_splicing" : "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/yellow_splicing",
                            
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
    
    if args.ris == -1:
        if not os.path.exists(os.path.join(base_save_path, 'aug_train.pt')):
            train_dataset = ad_aug_dataset(args.data_name, data_path_dict, 'train_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules,
                                        save=args.save, save_path=os.path.join(args.save_path, "train/good"))
            torch.save(train_dataset, os.path.join(base_save_path, 'aug_train.pt'))
            print("Train image saved")
        else:
            print("Train image already exists")

        if not os.path.exists(os.path.join(base_save_path, 'aug_val.pt')):
            val_dataset = ad_aug_dataset(args.data_name, data_path_dict, 'val_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules,
                                        save=args.save, save_path=os.path.join(args.save_path, "validation/good"))
            torch.save(val_dataset, os.path.join(base_save_path, 'aug_val.pt'))
            print("Valid image saved")
        else:
            print("Valid image already exists")

        if not os.path.exists(os.path.join(base_save_path, 'aug_test_good.pt')):
            test_good_dataset = ad_aug_dataset(args.data_name, data_path_dict, 'test_good_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules, 
                                        save=args.save, save_path=os.path.join(args.save_path, "test/good"))
            torch.save(test_good_dataset, os.path.join(base_save_path, 'aug_test_good.pt'))
            print("Test good image saved")
        else:
            print("Test good image already exists")
        
        
        if not os.path.exists(os.path.join(base_save_path, 'aug_test_sa.pt')):
            test_sa_dataset = ad_aug_dataset(args.data_name, data_path_dict, 'test_sa_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules,
                                        save=args.save, save_path=os.path.join(args.save_path, "test/structural_anomalies"))
                                        
            torch.save(test_sa_dataset, os.path.join(base_save_path, 'aug_test_sa.pt'))
            print("Test sa saved")
        
        else:
            print("Test sa image already exists")
        

        if not os.path.exists(os.path.join(base_save_path, 'aug_test_la.pt')):
            test_la_dataset = ad_aug_dataset(args.data_name, data_path_dict, 'test_la_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules,
                                        save=args.save, save_path=os.path.join(args.save_path, "test/logical_anomalies"))
            torch.save(test_la_dataset, os.path.join(base_save_path, 'aug_test_la.pt'))
            print("Test la saved")
        else:
            print("Test la image already exists")


    elif args.ris == 0:
        
        if not os.path.exists(os.path.join(base_save_path, 'train.pt')):
            from pdb import set_trace
            #set_trace()
            train_dataset = ad_obj_dataset(data_path_dict, 'train_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules, overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride,
                                        save=args.save, save_path=os.path.join(args.save_path, "train/good"))
            torch.save(train_dataset, os.path.join(base_save_path, 'train.pt'))
            print("Train image saved")
        else:
            print("Train image already exists")

        """
        if not os.path.exists(os.path.join(base_save_path, 'val.pt')):
            val_dataset = ad_obj_dataset(data_path_dict, 'val_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules, overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride,
                                        save=args.save, save_path=os.path.join(args.save_path, "validation/good"))
            torch.save(val_dataset, os.path.join(base_save_path, 'val.pt'))
            print("Valid image saved")
        else:
            print("Valid image already exists")
        """
        if not os.path.exists(os.path.join(base_save_path, 'test_good.pt')):
            test_good_dataset = ad_obj_dataset(data_path_dict, 'test_good_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules, overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride,
                                        save=args.save, save_path=os.path.join(args.save_path, "test/good"))
            torch.save(test_good_dataset, os.path.join(base_save_path, 'test_good.pt'))
            print("Test good image saved")
        else:
            print("Test good image already exists")
        
        """
        if not os.path.exists(os.path.join(base_save_path, 'test_sa.pt')):
            test_sa_dataset = ad_obj_dataset(data_path_dict, 'test_sa_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules, overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride)
            torch.save(test_sa_dataset, os.path.join(base_save_path, 'test_sa.pt'))
            print("Test sa saved")
        
        else:
            print("Test sa image already exists")
        """

        if not os.path.exists(os.path.join(base_save_path, 'test_la.pt')):
            test_la_dataset = ad_obj_dataset(data_path_dict, 'test_la_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules, overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride,
                                        save=args.save, save_path=os.path.join(args.save_path, "test/logical_anomalies"))
            torch.save(test_la_dataset, os.path.join(base_save_path, 'test_la.pt'))
            print("Test la saved")
        else:
            print("Test la image already exists")
    
    elif args.ris == 1:
        print("Referring Image Segmentation Start")
        if not os.path.exists(os.path.join(base_save_path, 'new_ris_train.pt')):
            train_dataset = ad_ris_dataset(data_path_dict, 'train_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules, sam_num_list=sam_num_list, sam_text_list=sam_text_list
                                        , overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride,
                                        save=args.save, save_path=os.path.join(args.save_path, "train/good"))
            torch.save(train_dataset, os.path.join(base_save_path, 'new_ris_train.pt'))
            print("Train image saved")
        else:
            print("Train image already exists")

        if not os.path.exists(os.path.join(base_save_path, 'new_ris_val.pt')):
            val_dataset = ad_ris_dataset(data_path_dict, 'val_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules, sam_num_list=sam_num_list, sam_text_list=sam_text_list
                                        , overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride,
                                        save=args.save, save_path=os.path.join(args.save_path, "validation/good"))
            torch.save(val_dataset, os.path.join(base_save_path, 'new_ris_val.pt'))
            print("Valid image saved")
        else:
            print("Valid image already exists")

        if not os.path.exists(os.path.join(base_save_path, 'new_ris_test_good.pt')):
            test_good_dataset = ad_ris_dataset(data_path_dict, 'test_good_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules, sam_num_list=sam_num_list, sam_text_list=sam_text_list
                                        , overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride,
                                        save=args.save, save_path=os.path.join(args.save_path, "test/good"))
            torch.save(test_good_dataset, os.path.join(base_save_path, 'new_ris_test_good.pt'))
            print("Test good image saved")
        else:
            print("Test good image already exists")
         
        """
        if not os.path.exists(os.path.join(base_save_path, 'ris_test_sa.pt')):
            test_sa_dataset = ad_ris_dataset(data_path_dict, 'test_sa_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules, overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride)
            torch.save(test_sa_dataset, os.path.join(base_save_path, 'ris_test_sa.pt'))
            print("Test sa saved")
        
        else:
            print("Test sa image already exists")
        """

        if not os.path.exists(os.path.join(base_save_path, 'new_ris_test_la.pt')):
            test_la_dataset = ad_ris_dataset(data_path_dict, 'test_la_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=object_names, target_objects_list=objects_for_rules, sam_num_list=sam_num_list, sam_text_list=sam_text_list
                                        , overlap_count=args.overlap_count, kernel_size=args.kernel_size, stride=args.stride,
                                        save=args.save, save_path=os.path.join(args.save_path, "test/logical_anomalies"))
            torch.save(test_la_dataset, os.path.join(base_save_path, 'new_ris_test_la.pt'))
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
    parser.add_argument('--model_name', default='ViT-B-32', type=str)   #ViT-g-14
    parser.add_argument('--pretrained_name', default='laion2b_s34b_b79k', type=str)     #laion2b_s12b_b42k
    parser.add_argument('--lora', default=-1, type=int)
    parser.add_argument('--default', default=False, type=bool)
    parser.add_argument('--foundation', default="CLIP", type=str)
    parser.add_argument('--detection', default=False, type=bool)
    parser.add_argument('--kernel_size', default=250, type=int)
    parser.add_argument('--stride', default=50, type=int)
    parser.add_argument('--overlap_count', default=7, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--save_path', default="./", type=str)
    parser.add_argument('--ris', default=0, type=int)
    args = parser.parse_args()

    print(args)

    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    set_random_seed(random_seed=args.seed)

    print(f"Torch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is loaded for process")

    

    main(args)
