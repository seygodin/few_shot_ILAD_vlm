import torch
import torch.utils.data
import os
import numpy as np
import time
from sklearn.metrics import roc_auc_score
import sys
from csv import writer
import random
from ad_clip.utils_save import get_rule, get_data_path, get_rule_tokens, get_object_info
from ad_clip.model import get_model
from ad_clip.data import ad_dataset, ad_obj_dataset
from ad_clip.loss import compute_negative_count_loss
from csv import writer
from datetime import datetime
import argparse
from ad_clip import my_clip
import loralib as lora
from distutils.util import strtobool
from torch.utils.data import Subset
from torch.nn import functional as F
import copy

def set_random_seed(random_seed=123):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # multi-GPU
    print(f"Random seed {random_seed} is selected.")


def save_model(args, model, performance_metric, epoch, tag, base_directory:str ="./pretrained"):
    # 현재 날짜를 yyyy-mm-dd 형식으로 가져옴
    date_str = datetime.now().strftime("%Y-%m-%d")
    # 날짜별 디렉토리 경로 생성
    directory = os.path.join(base_directory, date_str)

    # 디렉토리가 없으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 파일 이름에 성능 지표와 에포크 번호를 포함
    filename = f"{args.data_name}_model_epoch_{epoch}_metric_{performance_metric:.4f}.pt"
    filepath = os.path.join(directory, filename)

    torch.save(lora.lora_state_dict(model), filepath)   #lora save method
    #torch.save(model.state_dict(), filepath)           #general pytorch save method
    print(f"Model saved: {filepath}")

    return filepath

def evaluate(args, model, tokenizer, pairs, test_good_dataset, test_sa_dataset, test_la_dataset):
    model.eval()
    sa_img_pred, la_img_pred = [], []

    good_probs = []
    la_probs = []
    sa_probs = []
    text_feature_list = []

    #Generating Image features
    if args.detection != 1:
        good_img_features = model.encode_image(test_good_dataset[:len(test_good_dataset)])
        #sa_img_features = model.encode_image(test_sa_dataset[:len(test_sa_dataset)])
        la_img_features = model.encode_image(test_la_dataset[:len(test_la_dataset)])
    else:
        good_img_features = model.encode_image(test_good_dataset[:len(test_good_dataset)][0])
        #sa_img_features = model.encode_image(test_sa_dataset[:len(test_sa_dataset)][0])
        la_img_features = model.encode_image(test_la_dataset[:len(test_la_dataset)][0])

    #Generating Text features
    for rule_idx, pair in enumerate(pairs):
        if args.text == 1:
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

        if args.detection != 1:
            pass
        elif args.detection ==1 and args.ris != 1:
            if args.mask == 1:
                if args.double_encoder != 1:
                    maksed_good_img_features = model.encode_image(test_good_dataset[:len(test_good_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                    #maksed_sa_img_features = model.encode_image(test_sa_dataset[:len(test_sa_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                    maksed_la_img_features = model.encode_image(test_la_dataset[:len(test_la_dataset)][1][:,rule_idx,:,:,:].squeeze(1))

                else:
                    maksed_good_img_features = model.encode_mask_image(test_good_dataset[:len(test_good_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                    #maksed_sa_img_features = model.encode_image(test_sa_dataset[:len(test_sa_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
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

    if args.score == "mean":
        la_probs = torch.mean(la_probs, dim=1)
    elif args.score == "min":
        la_probs = torch.min(la_probs, dim=1).values
    elif args.score == "max":
        la_probs = torch.max(la_probs, dim=1).values
    elif args.score == "median":
        la_probs = torch.median(la_probs, dim=1).values
    

    #sa_img_pred = torch.cat((good_probs, sa_probs)).detach().cpu().numpy()
    la_img_pred = torch.cat((good_probs, la_probs)).detach().cpu().numpy()

    #sa_labels = [1 for i in range(len(good_probs))] + [0 for i in range(len(sa_probs))]
    #la_labels = [1 for i in range(len(good_probs))] + [0 for i in range(len(la_probs))]
    la_labels = [1 for i in range(len(good_probs))] + [0 for i in range(len(la_probs))]
    #sa_auc = roc_auc_score(y_true=sa_labels, y_score=sa_img_pred)
    la_auc = roc_auc_score(y_true=la_labels, y_score=la_img_pred)
    #print(f"Pred: {sa_img_pred}")
    #print(f"Label: {sa_labels}")

    """
    new_information = [sa_auc, la_auc, list(sa_img_pred), sa_labels, list(la_img_pred), la_labels]
    col_names = ['sa', 'la', 'sa_img_pred', 'sa_labels', 'la_img_pred', 'la_labels']
    file_name = 'record.csv'

    if not os.path.exists(file_name):
        with open(file_name, 'w') as record:
            writer_object = writer(record)
            writer_object.writerow(col_names)
            record.close()

    with open(file_name, 'a') as record:
        writer_object = writer(record)
        writer_object.writerow(new_information)
        record.close()
    """
    sa_auc = 0
    return sa_auc, la_auc


def valid(args, model, tokenizer, pairs, valid_dataloader):
    model.eval()
    loss_sum = 0
    
    if args.detection != 1:
        for batch, img in enumerate(valid_dataloader):
            total_loss = 0
            for pair in pairs:
                logits_per_image, logits_per_text, loss = model(img, pair, return_loss=True)
                total_loss += loss
            loss_sum+=total_loss.detach()

    elif args.detection==1 and args.ris != 1:
        for batch, img in enumerate(valid_dataloader):
            total_loss = 0
            raw_image, masked_image = img
            for rule_idx, pair in enumerate(pairs):

                if args.mask == 1:
                    global_image_feature = model.encode_image(raw_image)
                    masked_image_for_rule = masked_image[:,rule_idx,:,:,:].squeeze(1)

                    if args.double_encoder != 1:
                        masked_image_feature = model.encode_image(masked_image_for_rule)
                    else:
                        masked_image_feature = model.encode_mask_image(masked_image_for_rule)
                    image_feature = (global_image_feature + masked_image_feature)
                else:
                    image_feature = model.encode_image(raw_image)
                
                if args.text == 1:
                    local_text = tokenizer(args.texts_for_rules[rule_idx]).to(device)
                    global_text_feature = model.encode_text(pair)
                    local_text_feature = model.encode_text(local_text)
                    text_feature = (global_text_feature + local_text_feature)
                else:
                    text_feature = model.encode_text(pair)

                loss = model.clip_loss(image_feature, text_feature, model.clip.logit_scale.exp())
                total_loss += loss
            loss_sum+=total_loss.detach()

    if args.log == 1:
        print(f"Valid Loss: {loss_sum/len(valid_dataloader)}")
    return loss_sum



def train(args, model, tokenizer, pairs, train_dataloader, valid_dataloader, test_good_dataset, test_sa_dataset, test_la_dataset, optimizer, epoch=1, tag="default"):
    start = time.time()
    best_la_auc = 0

    for i in range(epoch):
        loss_sum = 0
        neg_loss_sum = 0

        
        if i % 10 == 0:
            val_loss = valid(args, model, tokenizer, pairs, valid_dataloader)

        com_la = 0
        sa_auc, la_auc = evaluate(args, model, tokenizer, pairs, test_good_dataset, test_sa_dataset, test_la_dataset)
        com_la += la_auc
        if args.log:
            print(f"[Test ROC-AUC]  -  SA: {sa_auc:.5f},  LA: {la_auc:.5f}")

        sa_auc, la_auc = evaluate(args, model, tokenizer, pairs, temp_test_good_dataset1, test_sa_dataset, temp_test_la_dataset1)
        com_la += la_auc

        if args.log:
            print(f"[Test ROC-AUC]  -  SA: {sa_auc:.5f},  LA: {la_auc:.5f}")

        sa_auc, la_auc = evaluate(args, model, tokenizer, pairs, temp_test_good_dataset2, test_sa_dataset, temp_test_la_dataset2)
        com_la += la_auc

        if args.log:
            print(f"[Test ROC-AUC]  -  SA: {sa_auc:.5f},  LA: {la_auc:.5f}")
            print(f"[Average Roc-AUC]: {com_la/3:.5f}")
        if la_auc > best_la_auc:
            if la_auc > 0.9 and (la_auc-best_la_auc) >= 0.01:
                #saved_path = save_model(args, model=model, performance_metric=la_auc, epoch=i, tag=tag, base_directory="./pretrained")
                pass
            best_la_auc = la_auc
        
        model.train()
        if args.detection != 1 and args.ris != 1:
            for idx, img in enumerate(train_dataloader):
                total_loss = 0
                for pair in pairs:
                    logits_per_image, logits_per_text, loss = model(img, pair, return_loss=True)
                    total_loss += loss
                
                if not args.default:
                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    pass

                loss_sum+=total_loss.detach()
            if args.log==1:
                print(f"Epoch: {i+1} | Loss: {loss_sum/len(train_dataloader)}")

        elif args.detection ==1 and args.ris != 1:
            for idx, img in enumerate(train_dataloader):
                raw_image, masked_image = img
                total_loss = 0
                total_neg_loss = 0
                global_image_feature = model.encode_image(raw_image)
                for rule_idx, pair in enumerate(pairs):
                    
                    if args.mask==1:
                        masked_image_for_rule = masked_image[:,rule_idx,:,:,:].squeeze(1)
                        if args.double_encoder != 1:
                            masked_image_feature = model.encode_image(masked_image_for_rule)
                        else:
                            masked_image_feature = model.encode_mask_image(masked_image_for_rule)
                        image_feature = (global_image_feature + masked_image_feature)
                    else:
                        image_feature = global_image_feature


                    if args.text ==1:
                        local_text = tokenizer(args.texts_for_rules[rule_idx]).to(device)
                        global_text_feature = model.encode_text(pair)
                        local_text_feature = model.encode_text(local_text)
                        text_feature = (global_text_feature + local_text_feature)
                    else:
                        text_feature = model.encode_text(pair)

                    loss = model.clip_loss(image_feature, text_feature, model.clip.logit_scale.exp())
                    
                    negative_loss = 0
                    if args.neg_loss != 0:  
                        num_img = image_feature.shape[0]
                        if num_img == 1:
                            negative_loss = compute_negative_count_loss(image_feature, text_feature[0], text_feature[1:])
                        else:
                            for num_img_idx in range(num_img):
                                negative_loss += compute_negative_count_loss(image_feature[num_img_idx], text_feature[0], text_feature[1:])
                            negative_loss /= num_img
                        
                    else:
                        negative_loss = 0
                        

                    total_loss += (loss + args.neg_loss * negative_loss)

                    total_neg_loss += negative_loss

                if not args.default:
                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                else:
                    pass

                loss_sum+=total_loss.detach().cpu().numpy()

                if args.neg_loss != 0:
                    neg_loss_sum += total_neg_loss.detach().cpu().numpy()
                else:
                    neg_loss_sum += 0

            if args.log == 1:
                print(f"Epoch: {i} | Loss: {loss_sum/len(train_dataloader)}  | Negative Loss: {neg_loss_sum/len(train_dataloader)}")
                
       
        
    end = time.time()

    print(f"Time consumption: {end-start}")
    print(f"Best LA AUC: {best_la_auc}")

import open_clip

def main(args):
    juice_bottles = ["banana_juice", "orange_juice", "cherry_juice"]
    splicing = ["blue_splicing", "red_splicing", "yellow_splicing"]

    if args.data_name in juice_bottles:
        juice_bottles.remove(args.data_name)
        sub_product_data_names = juice_bottles
    elif args.data_name in splicing:
        splicing.remove(args.data_name)
        sub_product_data_names = splicing
    print(f"Extra data: {sub_product_data_names}")
    object_names, objects_for_rules, texts_for_rules, _ = get_object_info(args.data_name, ris_info=False)
    args.object_names = object_names
    args.objects_for_rules = objects_for_rules
    args.texts_for_rules = texts_for_rules
    print(args)
    print("Loading model")
    my_model, preprocess, tokenizer, optimizer = get_model(args, model_name=args.model_name, pretrained_name=args.pretrained_name, default=args.default, double_encoder=args.double_encoder)
    my_model = my_model.to(device)
    #preprocess.transforms = preprocess.transforms[:4]

    print("Tokenizing rules")
    if args.num_rule == 0:
        rules_for_data = get_rule(data_name = args.data_name, rule_idxs="max")
    else:
        rules_for_data = get_rule(data_name = args.data_name, rule_idxs=[i for i in range(args.num_rule)])
    rule_token_pairs = get_rule_tokens(rules=rules_for_data, tokenizer=tokenizer, device=device)
    
    print("Generating dataloaders")
    data_path_dict = get_data_path(args.data_name, log_option=False)

    if args.detection != 1:
        train_dataset = ad_dataset(data_path_dict, 'train_path', preprocess=preprocess, level=3, device=device)
        val_dataset = ad_dataset(data_path_dict, 'val_path', preprocess=preprocess, level=3, device=device)
        test_good_dataset = ad_dataset(data_path_dict, 'test_good_path', preprocess=preprocess, level=3, device=device)
        test_sa_dataset = ad_dataset(data_path_dict, 'test_sa_path', preprocess=preprocess, level=3, device=device)
        test_la_dataset = ad_dataset(data_path_dict, 'test_la_path', preprocess=preprocess, level=3, device=device)

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
    
    else:
    
        if args.preprocess != 1:
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            tokenizer = open_clip.get_tokenizer('ViT-B-32')

            train_dataset = ad_obj_dataset(data_path_dict, 'train_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=args.object_names, target_objects_list=args.objects_for_rules, overlap_count=7, kernel_size=200, stride=50)
            
            val_dataset = ad_obj_dataset(data_path_dict, 'val_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=args.object_names, target_objects_list=args.objects_for_rules, overlap_count=7, kernel_size=200, stride=50)
            test_good_dataset = ad_obj_dataset(data_path_dict, 'test_good_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=args.object_names, target_objects_list=args.objects_for_rules, overlap_count=7, kernel_size=200, stride=50)
            test_sa_dataset = ad_obj_dataset(data_path_dict, 'test_sa_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=args.object_names, target_objects_list=args.objects_for_rules, overlap_count=7, kernel_size=200, stride=50)
            test_la_dataset = ad_obj_dataset(data_path_dict, 'test_la_path', model=model, preprocess=preprocess, tokenizer=tokenizer, device=device,
                                        object_names=args.object_names, target_objects_list=args.objects_for_rules, overlap_count=7, kernel_size=200, stride=50)
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

            if args.ris != 1:
                train_dataset = torch.load(os.path.join(base_save_path, "train.pt"))
                val_dataset = torch.load(os.path.join(base_save_path, 'val.pt'))
                test_good_dataset = torch.load(os.path.join(base_save_path, 'test_good.pt'))
                test_la_dataset = torch.load(os.path.join(base_save_path, 'test_la.pt'))
                test_sa_dataset = torch.load(os.path.join(base_save_path, 'test_sa.pt'))
            elif args.ris == 1:
                """
                train_dataset = MDataset(torch.load(os.path.join(base_save_path, "ris_train.pt")))
                val_dataset = MDataset(torch.load(os.path.join(base_save_path, 'ris_val.pt')))
                test_good_dataset = MDataset(torch.load(os.path.join(base_save_path, 'ris_test_good.pt')))
                test_la_dataset = MDataset(torch.load(os.path.join(base_save_path, 'ris_test_la.pt')))
                test_sa_dataset = MDataset(torch.load(os.path.join(base_save_path, 'ris_test_sa.pt')))
                """
                train_dataset = torch.load(os.path.join(base_save_path, "ris_train.pt"))
                val_dataset = torch.load(os.path.join(base_save_path, 'ris_val.pt'))
                test_good_dataset = torch.load(os.path.join(base_save_path, 'ris_test_good.pt'))
                test_la_dataset = torch.load(os.path.join(base_save_path, 'ris_test_la.pt'))
                test_sa_dataset = torch.load(os.path.join(base_save_path, 'ris_test_sa.pt'))

            if sub_product_data_names != []:
                temp_base_save_path1 = data_base_path_dict[sub_product_data_names[0]]
                global temp_test_good_dataset1 
                temp_test_good_dataset1 = torch.load(os.path.join(temp_base_save_path1, 'test_good.pt'))
                global temp_test_la_dataset1
                temp_test_la_dataset1 = torch.load(os.path.join(temp_base_save_path1, 'test_la.pt'))

                temp_base_save_path2 = data_base_path_dict[sub_product_data_names[1]]
                global temp_test_good_dataset2
                temp_test_good_dataset2 = torch.load(os.path.join(temp_base_save_path2, 'test_good.pt'))
                global temp_test_la_dataset2
                temp_test_la_dataset2 = torch.load(os.path.join(temp_base_save_path2, 'test_la.pt'))

        if args.few_shot ==1:
            sampled_indices = random.sample(range(len(train_dataset)), args.shot)
            train_dataset = Subset(train_dataset, sampled_indices)
        
        print(f"[Data Description]")
        print(f"- Train: {len(train_dataset)}")
        print(f"- Valid: {len(val_dataset)}")
        print(f"- Test: {len(test_good_dataset)+len(test_la_dataset)}\n")
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)#, drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=args.shuffle)#, drop_last=True)
        
    print("Train Start")
    if not args.ciriculam_learning:
        train(args, model=my_model, tokenizer=tokenizer, pairs=rule_token_pairs, train_dataloader=train_dataloader, valid_dataloader=val_dataloader, test_good_dataset=test_good_dataset, 
          test_sa_dataset=test_sa_dataset, test_la_dataset=test_la_dataset, optimizer=optimizer, epoch=args.epoch, tag=args.tag)
    else:
        for num_rule in range(1, len(rule_token_pairs)):
            print(f"=== [{num_rule-1} ciriculam] ===")
            temp_rule_token_pairs = rule_token_pairs[:num_rule]
            train(args, model=my_model, tokenizer=tokenizer, pairs=temp_rule_token_pairs, train_dataloader=train_dataloader, valid_dataloader=val_dataloader, test_good_dataset=test_good_dataset, 
                test_sa_dataset=test_sa_dataset, test_la_dataset=test_la_dataset, optimizer=optimizer, epoch=args.epoch, tag=args.tag)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--gpu_number', default="0", type=str)
    parser.add_argument('--data_name', default='breakfast', type=str)
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tag', default='default', type=str)
    parser.add_argument('--model_name', default='ViT-B-32', type=str)
    parser.add_argument('--pretrained_name', default='laion2b_s34b_b79k', type=str)
    parser.add_argument('--lora', default=1, type=int)
    parser.add_argument('--default', default=False, type=strtobool)
    parser.add_argument('--foundation', default="CLIP", type=str)
    parser.add_argument('--detection', default=False, type=strtobool)
    parser.add_argument('--preprocess', default=False, type=strtobool)
    parser.add_argument('--num_rule', default=0, type=int)          #Breakfast: 7, juice_bottle: 8, pushpin: 4, screw_bag:5, connector: 6
    parser.add_argument('--object_names', default=[], type=list)
    parser.add_argument('--objects_for_rules', default=[], type=list)
    parser.add_argument('--texts_for_rules', default=[], type=list)
    parser.add_argument('--mask', default=False, type=strtobool)
    parser.add_argument('--text', default=False, type=strtobool)
    parser.add_argument('--score', default="mean", type=str)
    parser.add_argument('--ciriculam_learning', default=False, type=strtobool)
    parser.add_argument('--few_shot', default=False, type=strtobool)
    parser.add_argument('--shot', default=5, type=int)
    parser.add_argument('--log', default=False, type=strtobool)
    parser.add_argument('--neg_loss', default=0, type=float)
    parser.add_argument('--double_encoder', default=False, type=strtobool)
    parser.add_argument('--pretrained_count_path', default=None, type=str)
    parser.add_argument('--ris', default=0, type=int)
    args = parser.parse_args()
    #--model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k

    
    #model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    if args.seed != 0:
        set_random_seed(random_seed=args.seed)
    else:
        set_random_seed(random_seed=random.choice([i for i in range(100000)]))

    print(f"Torch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is loaded for process")
    
    

    main(args)
    
