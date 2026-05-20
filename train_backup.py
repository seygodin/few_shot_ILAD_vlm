import torch
import torch.utils.data
import os
import numpy as np
import time
from sklearn.metrics import roc_auc_score
import sys
from csv import writer
import random
from ad_clip.utils_save import get_rule, get_data_path, get_rule_tokens
from ad_clip.model import get_model
from ad_clip.data import ad_dataset
from csv import writer
from datetime import datetime
import argparse

def set_random_seed(random_seed=123):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # multi-GPU
    print(f"Random seed {random_seed} is selected.")


def save_model(model, performance_metric, epoch, tag, base_directory:str ="./pretrained"):
    # 현재 날짜를 yyyy-mm-dd 형식으로 가져옴
    date_str = datetime.now().strftime("%Y-%m-%d")
    # 날짜별 디렉토리 경로 생성
    directory = os.path.join(base_directory, date_str)

    # 디렉토리가 없으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 파일 이름에 성능 지표와 에포크 번호를 포함
    filename = f"{tag}_model_epoch_{epoch}_metric_{performance_metric:.4f}.pt"
    filepath = os.path.join(directory, filename)

    torch.save(model.state_dict(), filepath)
    print(f"Model saved: {filepath}")

    return filepath

def evaluate(model, pairs, test_good_dataset, test_sa_dataset, test_la_dataset):
    model.eval()
    sa_img_pred, la_img_pred = [], []

    good_img_features = model.encode_image(test_good_dataset[:len(test_good_dataset)])
    sa_img_features = model.encode_image(test_sa_dataset[:len(test_sa_dataset)])
    la_img_features = model.encode_image(test_la_dataset[:len(test_la_dataset)])

    good_probs = []
    la_probs = []
    sa_probs = []

    for pair in pairs:
        text_features = model.encode_text(pair)

        good_img_features /= good_img_features.norm(dim=-1, keepdim=True)
        sa_img_features /= sa_img_features.norm(dim=-1, keepdim=True)
        la_img_features /= la_img_features.norm(dim=-1, keepdim=True)

        text_features /= text_features.norm(dim=-1, keepdim=True)

        good_text_probs = (100.0 * good_img_features @ text_features.T).softmax(dim=-1)
        good_probs.append(good_text_probs[:,0])

        sa_text_probs = (100.0 * sa_img_features @ text_features.T).softmax(dim=-1)
        sa_probs.append(sa_text_probs[:,0])

        la_text_probs = (100.0 * la_img_features @ text_features.T).softmax(dim=-1)
        la_probs.append(la_text_probs[:,0])

    good_probs = torch.stack(good_probs, dim=1)
    good_probs = torch.mean(good_probs, dim=1)

    sa_probs = torch.stack(sa_probs, dim=1)
    sa_probs = torch.mean(sa_probs, dim=1)

    #print(la_probs.shape)
    la_probs = torch.stack(la_probs, dim=1)
    la_probs = torch.mean(la_probs, dim=1)

    sa_img_pred = torch.cat((good_probs, sa_probs)).detach().cpu().numpy()
    la_img_pred = torch.cat((good_probs, la_probs)).detach().cpu().numpy()

    sa_labels = [1 for i in range(len(good_probs))] + [0 for i in range(len(sa_probs))]
    la_labels = [1 for i in range(len(good_probs))] + [0 for i in range(len(la_probs))]

    sa_auc = roc_auc_score(y_true=sa_labels, y_score=sa_img_pred)
    la_auc = roc_auc_score(y_true=la_labels, y_score=la_img_pred)
    #print(f"Pred: {sa_img_pred}")
    #print(f"Label: {sa_labels}")

    
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


    return sa_auc, la_auc


def valid(model, pairs, valid_dataloader):
    model.eval()
    loss_sum = 0
    for batch, img in enumerate(valid_dataloader):
        total_loss = 0
        for pair in pairs:
            logits_per_image, logits_per_text, loss = model(img, pair, return_loss=True)
            total_loss += loss
        loss_sum+=total_loss.detach()
    print(f"Valid Loss: {loss_sum/len(valid_dataloader)}")
    return loss_sum


def train(model, pairs, train_dataloader, valid_dataloader, test_good_dataset, test_sa_dataset, test_la_dataset, optimizer, epoch=1, tag="default"):
    start = time.time()
    best_la_auc = 0

    for i in range(epoch):
        loss_sum = 0

        if i % 10 == 0:
            val_loss = valid(model, pairs, valid_dataloader)

        sa_auc, la_auc = evaluate(model, pairs, test_good_dataset, test_sa_dataset, test_la_dataset)
        print(f"[Test ROC-AUC]  -  SA: {sa_auc:.5f},  LA: {la_auc:.5f}")
        if la_auc > best_la_auc:
            saved_path = save_model(model=model, performance_metric=la_auc, epoch=i, tag=tag, base_directory="./pretrained")
            best_la_auc = la_auc

        model.train()
        for idx, img in enumerate(train_dataloader):
            total_loss = 0
            for pair in pairs:
                logits_per_image, logits_per_text, loss = model(img, pair, return_loss=True)
                total_loss += loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_sum+=total_loss.detach()

        print(f"Epoch: {i} | Loss: {loss_sum/len(train_dataloader)}")
        
    end = time.time()

    print(f"Time consumption: {end-start}")
    print(f"Best LA AUC: {best_la_auc}")

def main(args):
    print("Loading model")
    my_model, preprocess, tokenizer, optimizer = get_model(args, model_name=args.model_name, pretrained_name=args.pretrained_name, default=args.default)
    my_model = my_model.to(device)
    #preprocess.transforms = preprocess.transforms[:4]

    print("Tokenizing rules")
    rules_for_data = get_rule(data_name = args.data_name, rule_idxs=[i for i in range(8)])
    rule_token_pairs = get_rule_tokens(rules=rules_for_data, tokenizer=tokenizer, device=device)

    print("Generating dataloaders")
    data_path_dict = get_data_path(args.data_name)
    train_dataset = ad_dataset(data_path_dict, 'train_path', preprocess=preprocess, level=3, device=device)
    val_dataset = ad_dataset(data_path_dict, 'val_path', preprocess=preprocess, level=3, device=device)
    test_good_dataset = ad_dataset(data_path_dict, 'test_good_path', preprocess=preprocess, level=3, device=device)
    test_sa_dataset = ad_dataset(data_path_dict, 'test_sa_path', preprocess=preprocess, level=3, device=device)
    test_la_dataset = ad_dataset(data_path_dict, 'test_la_path', preprocess=preprocess, level=3, device=device)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    print("Train Start")
    train(model=my_model, pairs=rule_token_pairs, train_dataloader=train_dataloader, valid_dataloader=val_dataloader, test_good_dataset=test_good_dataset, 
          test_sa_dataset=test_sa_dataset, test_la_dataset=test_la_dataset, optimizer=optimizer, epoch=args.epoch, tag=args.tag)
    

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
    parser.add_argument('--lora', default=-1, type=int)
    parser.add_argument('--default', default=False, type=bool)
    parser.add_argument('--foundation', default="CLIP", type=str)
    args = parser.parse_args()

    print(args)

    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    set_random_seed(random_seed=args.seed)

    print(f"Torch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is loaded for process")

    

    main(args)
    
