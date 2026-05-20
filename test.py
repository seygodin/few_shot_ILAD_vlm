import argparse
import torch
import os

import numpy as np
import random

from ad_clip.utils_save import get_rule, get_data_path, get_rule_tokens
from ad_clip.model import get_model
from ad_clip.data import ad_dataset

from sklearn.metrics import roc_auc_score, roc_curve

def set_random_seed(random_seed=123):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # multi-GPU
    print(f"Random seed {random_seed} is selected.")

def evaluate(args, model, pairs, test_good_dataset, test_sa_dataset, test_la_dataset, good_image_path, la_image_path):
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
    sa_probs = torch.stack(sa_probs, dim=1)
    la_probs = torch.stack(la_probs, dim=1)

    if args.eval_avg:
        good_probs = torch.mean(good_probs, dim=1)
        sa_probs = torch.mean(sa_probs, dim=1)
        la_probs = torch.mean(la_probs, dim=1)
    else:
        good_probs = torch.min(good_probs, dim=1).values
        sa_probs = torch.min(sa_probs, dim=1).values
        la_probs = torch.min(la_probs, dim=1).values


    sa_img_pred = torch.cat((good_probs, sa_probs)).detach().cpu().numpy()
    la_img_pred = torch.cat((good_probs, la_probs)).detach().cpu().numpy()

    sa_labels = [1 for i in range(len(good_probs))] + [0 for i in range(len(sa_probs))]
    la_labels = [1 for i in range(len(good_probs))] + [0 for i in range(len(la_probs))]

    sa_auc = roc_auc_score(y_true=sa_labels, y_score=sa_img_pred)
    la_auc = roc_auc_score(y_true=la_labels, y_score=la_img_pred)

    if args.detail:
        # ROC 곡선 계산
        fpr, tpr, thresholds = roc_curve(la_labels, la_img_pred)
        # Youden's J 통계를 최대화하는 임계값 찾기
        J = tpr - fpr
        optimal_idx = np.argmax(J)
        optimal_threshold = thresholds[optimal_idx]


        good_wrong, la_wrong =0, 0
        for i in range(len(good_probs)):
            img_path = good_image_path[i]
            prob = float(good_probs[i].detach().cpu().numpy())
            if prob >= optimal_threshold:
                print(f"{prob}  ->  Right")
            else:
                print(f"{prob}  ->  Wrong  --> {img_path[-20:]}")
                good_wrong+=1
        print("===================")
        for i in range(len(la_probs)):
            img_path = la_image_path[i]
            prob = float(la_probs[i].detach().cpu().numpy())

            if prob >= optimal_threshold:
                print(f"{prob}  ->  Wrong  --> {img_path[-20:]}")
                la_wrong+=1
            else:
                print(f"{prob}  ->  Right")
                
        print(f"GOOD case | wrong: {good_wrong}  / {len(good_probs)}")
        print(f"LA case | wrong: {la_wrong}  / {len(la_probs)}")

    return sa_auc, la_auc

def main(args):
    print("Generating model")
    my_model, preprocess, tokenizer, optimizer = get_model(args, model_name=args.model_name, pretrained_name=args.pretrained_name)

    try:
        my_model.load_state_dict(torch.load(args.model_path))
        print("Model load success")
    except:
        raise RuntimeError("Failed to load pretrained model")
    my_model = my_model.to(device)
    #preprocess.transforms = preprocess.transforms[:4]

    print("Tokenizing rules")
    rules_for_data = get_rule(data_name = args.data_name, rule_idxs=[i for i in range(8)])
    rule_token_pairs = get_rule_tokens(rules=rules_for_data, tokenizer=tokenizer, device=device)

    print("Generating dataloaders")
    data_path_dict = get_data_path(args.data_name)
    test_good_dataset = ad_dataset(data_path_dict, 'test_good_path', preprocess=preprocess, level=3, device=device)
    test_sa_dataset = ad_dataset(data_path_dict, 'test_sa_path', preprocess=preprocess, level=3, device=device)
    test_la_dataset = ad_dataset(data_path_dict, 'test_la_path', preprocess=preprocess, level=3, device=device)

    good_image_path = [path for path in data_path_dict['test_good_path']]
    la_image_path = [path for path in data_path_dict['test_la_path']]

    sa_auc, la_auc = evaluate(args, my_model, rule_token_pairs, test_good_dataset, test_sa_dataset, test_la_dataset, good_image_path, la_image_path)

    print(f"Logical Anomaly AUC score: {la_auc}")
    print(f"Structural Anomaly AUC score: {sa_auc}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu_number', default="0", type=str)
    parser.add_argument('--data_name', default='breakfast', type=str)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--tag', default='default', type=str)
    parser.add_argument('--model_name', default='ViT-B-32', type=str)
    parser.add_argument('--pretrained_name', default='laion2b_s34b_b79k', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument("--lr", default=1e-6, type=float)
    parser.add_argument("--detail", default=False, type=bool)
    parser.add_argument("--eval_avg", default=True, type=bool)
    args = parser.parse_args()

    print(args)

    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    set_random_seed(random_seed=args.seed)

    print(f"Torch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is loaded for process")

    

    main(args)
    