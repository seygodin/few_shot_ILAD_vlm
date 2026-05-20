import torch
import torch.utils.data
import os
import numpy as np
import time
from sklearn.metrics import roc_auc_score, roc_curve
import sys
from csv import writer
import random
from ad_clip.utils_save import get_rule, get_data_path, get_rule_tokens, get_object_info
from ad_clip.model import get_model
from ad_clip.data import ad_dataset, ad_obj_dataset
from csv import writer
from datetime import datetime
import argparse
from ad_clip import my_clip
import loralib as lora

from torch.utils.data import Subset
from torch.nn import functional as F

import open_clip

def set_random_seed(random_seed=123):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # multi-GPU
    print(f"Random seed {random_seed} is selected.")

def evaluate(args, model, tokenizer, pairs, test_good_dataset, test_sa_dataset, test_la_dataset):
    model.eval()
    sa_img_pred, la_img_pred = [], []

    good_probs = []
    la_probs = []
    sa_probs = []
    text_feature_list = []

    #Generating Image features
    if not args.detection:
        good_img_features = model.encode_image(test_good_dataset[:len(test_good_dataset)])
        #sa_img_features = model.encode_image(test_sa_dataset[:len(test_sa_dataset)])
        la_img_features = model.encode_image(test_la_dataset[:len(test_la_dataset)])
    else:
        good_img_features = model.encode_image(test_good_dataset[:len(test_good_dataset)][0])
        #sa_img_features = model.encode_image(test_sa_dataset[:len(test_sa_dataset)][0])
        la_img_features = model.encode_image(test_la_dataset[:len(test_la_dataset)][0])

    #Generating Text features
    for rule_idx, pair in enumerate(pairs):
        if args.text:
            local_text = tokenizer(args.texts_for_rules[rule_idx]).to(device)
            global_text_feature = model.encode_text(pair)
            local_text_feature = model.encode_text(local_text)
            text_features = (global_text_feature + local_text_feature)
            text_feature_list.append(text_features)
        else:
            text_features = model.encode_text(pair)
            text_feature_list.append(text_features)

    #Computing probablity
    for rule_idx, pair in enumerate(pairs):
        text_features = text_feature_list[rule_idx]

        if not args.detection:
            pass
        else:
            if args.mask:
                if not args.double_encoder:
                    maksed_good_img_features = model.encode_image(test_good_dataset[:len(test_good_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                    #maksed_sa_img_features = model.encode_image(test_sa_dataset[:len(test_sa_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                    maksed_la_img_features = model.encode_image(test_la_dataset[:len(test_la_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                
                else:
                    maksed_good_img_features = model.encode_mask_image(test_good_dataset[:len(test_good_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                    #maksed_sa_img_features = model.encode_mask_image(test_sa_dataset[:len(test_sa_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                    maksed_la_img_features = model.encode_mask_image(test_la_dataset[:len(test_la_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                good_img_features = (good_img_features + maksed_good_img_features)
                #sa_img_features = (sa_img_features + maksed_sa_img_features)/2
                la_img_features = (la_img_features + maksed_la_img_features)
            else:
                pass

        good_img_features /= good_img_features.norm(dim=-1, keepdim=True)
        #sa_img_features /= sa_img_features.norm(dim=-1, keepdim=True)
        la_img_features /= la_img_features.norm(dim=-1, keepdim=True)

        text_features /= text_features.norm(dim=-1, keepdim=True)

        
        #Sample에 대한 good일 확률 계산. positive rule에 대한 similarity is considered as good sample probablity
        good_text_probs = (100.0 * good_img_features @ text_features.T).softmax(dim=-1)
        good_probs.append(good_text_probs[:,0])

        #sa_text_probs = (100.0 * sa_img_features @ text_features.T).softmax(dim=-1)
        #sa_probs.append(sa_text_probs[:,0])

        la_text_probs = (100.0 * la_img_features @ text_features.T).softmax(dim=-1)
        la_probs.append(la_text_probs[:,0])

    good_probs = torch.stack(good_probs, dim=1)

    if args.score == "mean":
        good_probs = torch.mean(good_probs, dim=1)
    elif args.score == "min":
        good_probs = torch.min(good_probs, dim=1).values
    elif args.score == "max":
        good_probs = torch.max(good_probs, dim=1).values
    elif args.score == "median":
        good_probs = torch.median(good_probs, dim=1).values

    #sa_probs = torch.stack(sa_probs, dim=1)
    #sa_probs = torch.mean(sa_probs, dim=1)

    #print(la_probs.shape)
    la_probs = torch.stack(la_probs, dim=1)
    
    good_prediction_record = good_probs.detach()
    la_prediction_record = la_probs.detach()

    if args.score == "mean":
        la_probs = torch.mean(la_probs, dim=1)
    elif args.score == "min":
        la_probs = torch.min(la_probs, dim=1).values
    elif args.score == "max":
        la_probs = torch.max(la_probs, dim=1).values
    elif args.score == "median":
        la_probs = torch.median(la_probs, dim=1).values
    
    la_img_pred = torch.cat((good_probs, la_probs)).detach().cpu().numpy()

    la_labels = [1 for i in range(len(good_probs))] + [0 for i in range(len(la_probs))]
    la_auc = roc_auc_score(y_true=la_labels, y_score=la_img_pred)

    fpr, tpr, threshold = roc_curve(y_true=la_labels, y_score=la_img_pred)

    correct = 0
    for img_idx in range(len(test_good_dataset)):
        test_img_path = test_good_dataset.img_paths[img_idx]
        score = la_img_pred[img_idx]
        
        if score >= threshold:
            correct+=1
            print(f"{test_img_path}  ->  Right")
        else:
            print(f"{test_img_path}  ->  Wrong")
    
        print(f"{good_prediction_record[img_idx]}  | Threshold: {threshold}\n")

    num_good = len(test_good_dataset)
    
    for img_idx in range(len(test_la_dataset)):
        test_img_path = test_la_dataset.img_paths[num_good+img_idx]
        score = la_img_pred[img_idx]
        
        if score >= threshold:     
            print(f"{test_img_path}  ->  Wrong")
        else:
            correct+=1
            print(f"{test_img_path}  ->  Right")
            
        print(f"{la_prediction_record[img_idx]}  | Threshold: {threshold}\n")


    print(f"Accuracy: {correct} / {len(test_good_dataset) + len(test_la_dataset)} | {correct/(len(test_good_dataset) + len(test_la_dataset))}")    
    sa_auc = 0
    return sa_auc, la_auc

def main(args):
    
    object_names, objects_for_rules, texts_for_rules = get_object_info(args.data_name)
    args.object_names = object_names
    args.objects_for_rules = objects_for_rules
    args.texts_for_rules = texts_for_rules
    print(args)

    #Load pre-trained CLIP weights
    my_model, preprocess, tokenizer, optimizer = get_model(args, model_name=args.model_name, pretrained_name=args.pretrained_name, default=args.default)

    #Load trained optimal lora weights
    my_model.load_state_dict(torch.load(args.model_path), strict=False)

    my_model = my_model.to(device)

    rules_for_data = get_rule(data_name = args.data_name, rule_idxs=[i for i in range(args.num_rule)])
    rule_token_pairs = get_rule_tokens(rules=rules_for_data, tokenizer=tokenizer, device=device)
    
    data_path_dict = get_data_path(args.data_name, log_option=False)

    if not args.detection:
        test_good_dataset = ad_dataset(data_path_dict, 'test_good_path', preprocess=preprocess, level=3, device=device)
        test_sa_dataset = ad_dataset(data_path_dict, 'test_sa_path', preprocess=preprocess, level=3, device=device)
        test_la_dataset = ad_dataset(data_path_dict, 'test_la_path', preprocess=preprocess, level=3, device=device)

    else:
    
        if not args.preprocess:
            sys.exit("Beforing evaluating model, please run make_detection.py and train.py. \nYou must have pretrained model and preprocessed dataloader.")
        else:
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

            test_good_dataset = torch.load(os.path.join(base_save_path, 'test_good.pt'))
            test_la_dataset = torch.load(os.path.join(base_save_path, 'test_la.pt'))
            test_sa_dataset = torch.load(os.path.join(base_save_path, 'test_sa.pt'))
        
    print(f"[Data Description]")
    print(f"- Test Good: {len(test_good_dataset)}")
    print(f"- Test   LA:{len(test_la_dataset)}\n")

    evaluate(args, model=my_model, tokenizer=tokenizer, pairs=rule_token_pairs, test_good_dataset=test_good_dataset, test_sa_dataset=test_sa_dataset, test_la_dataset=test_la_dataset)
        
    

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
    parser.add_argument('--model_name', default='ViT-B-32', type=str)
    parser.add_argument('--pretrained_name', default='laion2b_s34b_b79k', type=str)
    parser.add_argument('--lora', default=1, type=int)
    parser.add_argument('--default', default=False, type=bool)
    parser.add_argument('--foundation', default="CLIP", type=str)
    parser.add_argument('--detection', default=False, type=bool)
    parser.add_argument('--preprocess', default=False, type=bool)
    parser.add_argument('--num_rule', default=4, type=int)          #Breakfast: 7, juice_bottle: 8, pushpin: 4, screw_bag:5, connector: 6
    parser.add_argument('--object_names', default=[], type=list)
    parser.add_argument('--objects_for_rules', default=[], type=list)
    parser.add_argument('--texts_for_rules', default=[], type=list)
    parser.add_argument('--mask', default=False, type=bool)
    parser.add_argument('--text', default=False, type=bool)
    parser.add_argument('--score', default="mean", type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--double_encoder', default=False, type=bool)
    args = parser.parse_args()
    #--model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k

    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    set_random_seed(random_seed=args.seed)

    print(f"Torch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is loaded for process")
    

    main(args)