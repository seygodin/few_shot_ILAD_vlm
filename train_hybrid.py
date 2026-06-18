import torch
import torch.utils.data
import os
import numpy as np
import time
from sklearn.metrics import roc_auc_score
import sys
import csv
import random
from ad_clip.utils_save import get_rule, get_data_path, get_rule_tokens, get_object_info
from ad_clip.model import get_model
from ad_clip.data import ad_dataset, ad_obj_dataset
from ad_clip.loss import compute_negative_count_loss
from datetime import datetime
import argparse
from ad_clip import my_clip
from ad_clip import loralib as lora
from distutils.util import strtobool
from torch.utils.data import Subset
from torch.nn import functional as F
import copy
from PIL import Image
import cv2
from sklearn.metrics import precision_recall_curve, auc
from pdb import set_trace

def calculate_auprc_and_optimal_f1(true_label, pred_score):
    # Precision-Recall 곡선 계산
    precision, recall, thresholds = precision_recall_curve(true_label, pred_score)
    # AUPRC 계산
    auprc = auc(recall, precision)
    
    # 분모 계산 및 0 처리
    denom = precision[:-1] + recall[:-1]
    # 분모가 0인 위치를 찾습니다.
    zero_denom = denom == 0
    # F1-스코어 배열을 초기화합니다.
    f1_scores = np.zeros_like(denom)
    # 분모가 0이 아닌 위치에서 F1-스코어를 계산합니다.
    f1_scores[~zero_denom] = 2 * (precision[:-1][~zero_denom] * recall[:-1][~zero_denom]) / denom[~zero_denom]
    # 분모가 0인 위치의 F1-스코어는 이미 0으로 설정되어 있습니다.
    
    # 최적의 F1-스코어 인덱스 찾기
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    return auprc, optimal_f1
    
meta_projecter = torch.nn.Linear(512+4, 512)
meta_projecter.train()


DATA_BASE_PATH_DICT = {
    "breakfast": "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/breakfast_box",
    "juice_bot": "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/juice_bottle",
    "pushpins": "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/pushpins",
    "screw_bag": "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/screw_bag",
    "splicing": "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/splicing_connectors",
    "banana_juice": "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/banana_juice",
    "cherry_juice": "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/cherry_juice",
    "orange_juice": "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/orange_juice",
    "blue_splicing": "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/blue_splicing",
    "red_splicing": "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/red_splicing",
    "yellow_splicing": "/data3/seungeon/data/image/mvtec_loco_anomaly_detection/yellow_splicing",
    "pcb1": "/data3/seungeon/data/image/visa/pcb1",
    "pcb2": "/data3/seungeon/data/image/visa/pcb2",
    "pcb3": "/data3/seungeon/data/image/visa/pcb3",
    "pcb4": "/data3/seungeon/data/image/visa/pcb4",
    "cable": "/data3/seungeon/data/image/mvtec_ad/cable",
    "capsule": "/data3/seungeon/data/image/mvtec_ad/capsule",
    "transistor": "/data3/seungeon/data/image/mvtec_ad/transistor",
    "cable_all": "/data/seungeon/orig/cable_all",
    "capsule_all": "/data/seungeon/orig/capsule_all",
    "transistor_all": "/data/seungeon/orig/transistor_all",
}


HYBRID_CLASS_NAME_BY_PRODUCT = {
    "breakfast": "breakfast_box",
    "juice_bot": "juice_bottle",
    "pushpins": "pushpins",
    "screw_bag": "screw_bag",
    "splicing": "splicing_connectors",
    "banana_juice": "juice_bottle",
    "cherry_juice": "juice_bottle",
    "orange_juice": "juice_bottle",
    "blue_splicing": "splicing_connectors",
    "red_splicing": "splicing_connectors",
    "yellow_splicing": "splicing_connectors",
    "pcb1": "pcb1",
    "pcb2": "pcb2",
    "pcb3": "pcb3",
    "pcb4": "pcb4",
    "cable": "cable",
    "capsule": "capsule",
    "transistor": "transistor",
    "cable_all": "cable_all",
    "capsule_all": "capsule_all",
    "transistor_all": "transistor_all",
}

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


def append_experiment_result(args, result_dict, hybrid_only_result=None):
    log_path = args.result_log_path
    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir and (not os.path.exists(log_dir)):
        os.makedirs(log_dir, exist_ok=True)

    field_names = [
        "timestamp",
        "data_name",
        "tag",
        "model_name",
        "pretrained_name",
        "seed",
        "mask",
        "text",
        "double_encoder",
        "few_shot",
        "shot",
        "hybrid_model",
        "hybrid_weight",
        "beta_mask",
        "beta_text",
        "pos_rule_type",
        "neg_rule_type",
        "num_negative",
        "best_la_auc",
        "best_la_aupc",
        "best_la_f1",
        "hybrid_only_la_auc",
        "hybrid_only_la_auprc",
        "hybrid_only_la_f1",
        "epochs_target",
        "epochs_executed",
        "early_stop_k",
        "early_stopped",
        "train_seconds",
    ]

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_name": args.data_name,
        "tag": args.tag,
        "model_name": args.model_name,
        "pretrained_name": args.pretrained_name,
        "seed": args.seed,
        "mask": int(args.mask),
        "text": int(args.text),
        "double_encoder": int(args.double_encoder),
        "few_shot": int(args.few_shot),
        "shot": args.shot,
        "hybrid_model": args.hybrid_model,
        "hybrid_weight": args.hybrid_weight,
        "beta_mask": args.beta_mask,
        "beta_text": args.beta_text,
        "pos_rule_type": args.pos_rule_type,
        "neg_rule_type": args.neg_rule_type,
        "num_negative": args.num_negative,
        "best_la_auc": result_dict["best_la_auc"],
        "best_la_aupc": result_dict["best_la_aupc"],
        "best_la_f1": result_dict["best_la_f1"],
        "hybrid_only_la_auc": "" if hybrid_only_result is None else hybrid_only_result["hybrid_only_la_auc"],
        "hybrid_only_la_auprc": "" if hybrid_only_result is None else hybrid_only_result["hybrid_only_la_auprc"],
        "hybrid_only_la_f1": "" if hybrid_only_result is None else hybrid_only_result["hybrid_only_la_f1"],
        "epochs_target": result_dict["epochs_target"],
        "epochs_executed": result_dict["epochs_executed"],
        "early_stop_k": result_dict["early_stop_k"],
        "early_stopped": int(result_dict["early_stopped"]),
        "train_seconds": result_dict["train_seconds"],
    }

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        csv_writer = csv.DictWriter(f, fieldnames=field_names)
        if not file_exists:
            csv_writer.writeheader()
        csv_writer.writerow(row)

    print(f"[Result] Experiment result appended to {log_path}")

def _iter_batches(values, batch_size):
    for start_idx in range(0, len(values), batch_size):
        yield values[start_idx : start_idx + batch_size]


def _dataset_img_paths(dataset):
    if isinstance(dataset, Subset):
        if not hasattr(dataset.dataset, "img_paths"):
            raise AttributeError("Subset base dataset does not have img_paths.")
        return [dataset.dataset.img_paths[idx] for idx in dataset.indices]

    if not hasattr(dataset, "img_paths"):
        raise AttributeError("Dataset does not have img_paths.")
    return list(dataset.img_paths)


def _load_image_batch(image_paths, transform, device):
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (1024, 1024))

        img = Image.fromarray(img)
        images.append(transform(img))
        
        # with Image.open(image_path) as img:
        #     img = img.resize((1024, 1024))
        #     images.append(transform(img.convert("RGB")))
    
    return torch.stack(images, dim=0).to(device)


def _anomaly_scores_to_normal_probs(anomaly_scores):
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float32)
    if anomaly_scores.size == 0:
        return anomaly_scores

    score_min = float(np.min(anomaly_scores))
    score_max = float(np.max(anomaly_scores))
    if abs(score_max - score_min) < 1e-12:
        return np.full_like(anomaly_scores, fill_value=0.5)

    normalized_anomaly = (anomaly_scores - score_min) / (score_max - score_min)
    return 1.0 - normalized_anomaly


def _resolve_hybrid_class_name(args):
    if args.hybrid_class_name != "":
        return args.hybrid_class_name

    if args.data_name in HYBRID_CLASS_NAME_BY_PRODUCT:
        return HYBRID_CLASS_NAME_BY_PRODUCT[args.data_name]

    if args.data_name in DATA_BASE_PATH_DICT:
        return os.path.basename(DATA_BASE_PATH_DICT[args.data_name])

    return args.data_name


def _import_winclip_module():
    import importlib

    winclip_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid", "winclip")
    if winclip_root not in sys.path:
        sys.path.insert(0, winclip_root)
    return importlib.import_module("reproduce_WinCLIP")


def build_hybrid_score_bank(args, train_dataset, test_good_dataset, test_la_dataset, device):
    if args.hybrid_model == "none":
        return None

    train_paths = _dataset_img_paths(train_dataset)
    test_good_paths = _dataset_img_paths(test_good_dataset)
    test_la_paths = _dataset_img_paths(test_la_dataset)
    eval_paths = test_good_paths + test_la_paths

    if len(train_paths) == 0:
        raise ValueError("Train dataset is empty, cannot build hybrid score bank.")

    class_name = _resolve_hybrid_class_name(args)
    if args.log:
        print(
            f"[Hybrid] method={args.hybrid_model}, class_name={class_name}, "
            f"train={len(train_paths)}, eval={len(eval_paths)}, base_path={args.data_base_path}"
        )

    if args.hybrid_model == "winclip":
        #set_trace()
        winclip = _import_winclip_module()

        hybrid_model = winclip.CLIP_AD(args.hybrid_backbone).to(device)
        #hybrid_model.half()
        hybrid_model.eval()

        preprocess = hybrid_model.preprocess
        preprocess.transforms[0] = winclip.transforms.Resize(
            size=(args.hybrid_img_resize, args.hybrid_img_resize),
            interpolation=winclip.transforms.InterpolationMode.BICUBIC,
            max_size=None,
            antialias=None,
        )
        preprocess.transforms[1] = winclip.transforms.CenterCrop(size=(args.hybrid_img_cropsize, args.hybrid_img_cropsize))

        patch_size = 16
        obj_list = [class_name]
        normal_text_feature_bank, abnormal_text_feature_bank = winclip.prepare_text_future(hybrid_model, obj_list)

        large_memory = {class_name: []}
        mid_memory = {class_name: []}
        patch_memory = {class_name: []}

        with torch.no_grad():
            for batch_paths in _iter_batches(train_paths, args.hybrid_batch_size):
                batch_images = _load_image_batch(batch_paths, preprocess, device)
                large_tokens, mid_tokens, patch_tokens, _, _, _ = hybrid_model.encode_image(batch_images, patch_size)

                for large_token, mid_token, patch_token in zip(large_tokens, mid_tokens, patch_tokens):
                    large_memory[class_name].append(large_token)
                    mid_memory[class_name].append(mid_token)
                    patch_memory[class_name].append(patch_token)

        large_memory[class_name] = torch.cat(large_memory[class_name], dim=0)
        mid_memory[class_name] = torch.cat(mid_memory[class_name], dim=0)
        patch_memory[class_name] = torch.cat(patch_memory[class_name], dim=0)

        anomaly_scores = []
        with torch.no_grad():
            for batch_paths in _iter_batches(eval_paths, args.hybrid_batch_size):
                batch_images = _load_image_batch(batch_paths, preprocess, device)
                batch_size, _, h, w = batch_images.shape
                cls_names = [class_name for _ in range(batch_size)]
                cls_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

                average_normal_features = normal_text_feature_bank[cls_ids]
                average_abnormal_features = abnormal_text_feature_bank[cls_ids]
                text_features = torch.cat((average_normal_features, average_abnormal_features), dim=1).permute(0, 2, 1)

                large_tokens, mid_tokens, patch_tokens, class_tokens, large_scale, mid_scale = hybrid_model.encode_image(
                    batch_images, patch_size
                )

                zscore = winclip.compute_score(class_tokens, text_features)
                z0score = zscore[:, 0, 1]

                large_similarity = winclip.compute_sim(large_tokens, text_features)[:, :, 1]
                mid_similarity = winclip.compute_sim(mid_tokens, text_features)[:, :, 1]

                large_scale_score = winclip.harmonic_aggregation((batch_size, h // patch_size, w // patch_size), large_similarity, large_scale)
                mid_scale_score = winclip.harmonic_aggregation((batch_size, h // patch_size, w // patch_size), mid_similarity, mid_scale)

                few_large = winclip.few_shot(large_memory, large_tokens, cls_names)
                few_mid = winclip.few_shot(mid_memory, mid_tokens, cls_names)
                few_patch = winclip.few_shot(patch_memory, patch_tokens, cls_names)

                few_large = winclip.harmonic_aggregation((batch_size, h // patch_size, w // patch_size), few_large, large_scale).to(device)
                few_mid = winclip.harmonic_aggregation((batch_size, h // patch_size, w // patch_size), few_mid, mid_scale).to(device)
                few_patch = few_patch.reshape((batch_size, h // patch_size, w // patch_size)).to(device)

                few_shot_score = torch.nan_to_num((few_large + few_mid + few_patch) / 3.0, nan=0.0, posinf=0.0, neginf=0.0)
                z0score = (z0score + torch.max(torch.max(few_shot_score, dim=1)[0], dim=1)[0]) / 2.0
                anomaly_scores.extend(z0score.detach().cpu().numpy().tolist())

    elif args.hybrid_model == "promptad":
        from hybrid.PromptAD.PromptAD import PromptAD, TripletLoss

        try:
            hybrid_model = PromptAD(
                out_size_h=args.hybrid_resolution,
                out_size_w=args.hybrid_resolution,
                device=device,
                backbone=args.hybrid_backbone,
                pretrained_dataset=args.hybrid_pretrained_dataset,
                n_ctx=args.hybrid_promptad_n_ctx,
                n_pro=args.hybrid_promptad_n_pro,
                n_ctx_ab=args.hybrid_promptad_n_ctx_ab,
                n_pro_ab=args.hybrid_promptad_n_pro_ab,
                class_name=class_name,
                k_shot=max(1, len(train_paths)),
                img_resize=args.hybrid_img_resize,
                img_cropsize=args.hybrid_img_cropsize,
            ).to(device)
        except KeyError as err:
            raise ValueError(
                f"PromptAD class prompt for '{class_name}' is not defined. Set --hybrid_class_name to a supported class."
            ) from err

        hybrid_model.eval_mode()

        feature_map1_list = []
        feature_map2_list = []
        with torch.no_grad():
            for batch_paths in _iter_batches(train_paths, args.hybrid_batch_size):
                batch_images = _load_image_batch(batch_paths, hybrid_model.transform, device)
                _, _, feature_map1, feature_map2 = hybrid_model.encode_image(batch_images)
                feature_map1_list.append(feature_map1)
                feature_map2_list.append(feature_map2)
        hybrid_model.build_image_feature_gallery(
            torch.cat(feature_map1_list, dim=0),
            torch.cat(feature_map2_list, dim=0),
        )

        optimizer = torch.optim.SGD(
            hybrid_model.prompt_learner.parameters(),
            lr=args.hybrid_promptad_lr,
            momentum=args.hybrid_promptad_momentum,
            weight_decay=args.hybrid_promptad_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.hybrid_promptad_epoch, 1),
            eta_min=args.hybrid_promptad_eta_min,
        )
        criterion = torch.nn.CrossEntropyLoss().to(device)
        criterion_triplet = TripletLoss(margin=0.0)

        shuffled_train_paths = list(train_paths)
        for _ in range(args.hybrid_promptad_epoch):
            random.shuffle(shuffled_train_paths)
            for batch_paths in _iter_batches(shuffled_train_paths, args.hybrid_batch_size):
                batch_images = _load_image_batch(batch_paths, hybrid_model.transform, device)

                normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = hybrid_model.prompt_learner()

                optimizer.zero_grad()

                normal_text_features = hybrid_model.encode_text_embedding(
                    normal_text_prompt, hybrid_model.tokenized_normal_prompts
                )
                abnormal_text_features_handle = hybrid_model.encode_text_embedding(
                    abnormal_text_prompt_handle, hybrid_model.tokenized_abnormal_prompts_handle
                )
                abnormal_text_features_learned = hybrid_model.encode_text_embedding(
                    abnormal_text_prompt_learned, hybrid_model.tokenized_abnormal_prompts_learned
                )
                abnormal_text_features = torch.cat(
                    [abnormal_text_features_handle, abnormal_text_features_learned], dim=0
                )

                mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
                mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)
                loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0

                cls_feature, _, _, _ = hybrid_model.encode_image(batch_images)

                normal_text_anchor = normal_text_features.mean(dim=0).unsqueeze(0)
                normal_text_anchor = normal_text_anchor / normal_text_anchor.norm(dim=-1, keepdim=True)

                abnormal_text_anchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
                abnormal_text_anchor = abnormal_text_anchor / abnormal_text_anchor.norm(dim=-1, keepdim=True)
                abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)

                l_pos = torch.einsum("nc,cm->nm", cls_feature, normal_text_anchor.transpose(0, 1))
                l_neg_v2t = torch.einsum("nc,cm->nm", cls_feature, abnormal_text_features.transpose(0, 1))

                logit_scale = hybrid_model.model.logit_scale
                
                if hybrid_model.precision == "fp16":
                    logit_scale = logit_scale.half()

                logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale
                target_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)

                loss_v2t = criterion(logits_v2t, target_v2t)
                loss_triplet = criterion_triplet(cls_feature, normal_text_anchor, abnormal_text_anchor)

                loss = loss_v2t + loss_triplet + loss_match_abnormal * args.hybrid_promptad_lambda1
                loss.backward()
                optimizer.step()

            scheduler.step()
            hybrid_model.build_text_feature_gallery()

        if args.hybrid_promptad_epoch == 0:
            hybrid_model.build_text_feature_gallery()

        anomaly_scores = []
        hybrid_model.eval_mode()
        with torch.no_grad():
            for batch_paths in _iter_batches(eval_paths, args.hybrid_batch_size):
                batch_images = _load_image_batch(batch_paths, hybrid_model.transform, device)
                score_imgs, _ = hybrid_model(batch_images, "cls")
                anomaly_scores.extend(float(score_img) for score_img in score_imgs)

    else:
        raise ValueError(f"Unsupported hybrid_model: {args.hybrid_model}")


    # normal_probs = _anomaly_scores_to_normal_probs(anomaly_scores)
    # set_trace()
    inversed_probs = 1 - np.asarray(anomaly_scores, dtype=np.float32)
    # inverted_probs = 1.0 - normal_probs
    # return {path: float(prob) for path, prob in zip(eval_paths, inverted_probs)}

    return {path: float(prob) for path, prob in zip(eval_paths, inversed_probs)}


def evaluate_hybrid_only(test_good_dataset, test_la_dataset, hybrid_score_bank):
    if hybrid_score_bank is None:
        return None

    test_good_paths = _dataset_img_paths(test_good_dataset)
    test_la_paths = _dataset_img_paths(test_la_dataset)

    if len(test_good_paths) == 0 or len(test_la_paths) == 0:
        return None

    hybrid_good_prob = np.array([hybrid_score_bank.get(path, 0.5) for path in test_good_paths], dtype=np.float32)
    hybrid_la_prob = np.array([hybrid_score_bank.get(path, 0.5) for path in test_la_paths], dtype=np.float32)
    hybrid_only_pred = np.concatenate((hybrid_good_prob, hybrid_la_prob))
    la_labels = [1 for _ in range(len(hybrid_good_prob))] + [0 for _ in range(len(hybrid_la_prob))]

    hybrid_only_la_auc = roc_auc_score(y_true=la_labels, y_score=hybrid_only_pred)
    hybrid_only_la_auprc, hybrid_only_la_f1 = calculate_auprc_and_optimal_f1(
        true_label=la_labels,
        pred_score=hybrid_only_pred,
    )

    return {
        "hybrid_only_la_auc": float(hybrid_only_la_auc),
        "hybrid_only_la_auprc": float(hybrid_only_la_auprc),
        "hybrid_only_la_f1": float(hybrid_only_la_f1),
    }


def evaluate(args, model, tokenizer, pairs, test_good_dataset, test_sa_dataset, test_la_dataset, hybrid_score_bank=None):

    model.eval()
    sa_img_pred, la_img_pred = [], []

    good_probs = []
    la_probs = []
    sa_probs = []
    text_feature_list = []
    
    #set_trace()
    #Generating Image features
    if args.detection != 1:
        global_good_img_features = model.encode_image(test_good_dataset[:len(test_good_dataset)])
        #sa_img_features = model.encode_image(test_sa_dataset[:len(test_sa_dataset)])
        global_la_img_features = model.encode_image(test_la_dataset[:len(test_la_dataset)])
    else:
        global_good_img_features = model.encode_image(test_good_dataset[:len(test_good_dataset)][0])
        #sa_img_features = model.encode_image(test_sa_dataset[:len(test_sa_dataset)][0])
        global_la_img_features = model.encode_image(test_la_dataset[:len(test_la_dataset)][0])

    #Generating Text features
    for rule_idx, pair in enumerate(pairs):
        if args.text == 1:
            local_text = tokenizer(args.texts_for_rules[rule_idx]).to(device)
            global_text_feature = model.encode_text(pair)
            local_text_feature = model.encode_text(local_text)
            text_features = ((1.0 - args.beta_text) * global_text_feature + args.beta_text * local_text_feature)
            text_feature_list.append(text_features)
        else:
            text_features = model.encode_text(pair)
            text_feature_list.append(text_features)

    #Computing probablity
    for rule_idx, pair in enumerate(pairs):
        text_features = text_feature_list[rule_idx]

        if args.rule_select != "":
            if rule_idx +1 not in args.rule_list:
                continue


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
                good_img_features = ((1.0 - args.beta_mask) * global_good_img_features + args.beta_mask * maksed_good_img_features)
                #sa_img_features = (sa_img_features + maksed_sa_img_features)/2
                la_img_features = ((1.0 - args.beta_mask) * global_la_img_features + args.beta_mask * maksed_la_img_features)
            else:
                good_img_features = global_good_img_features
                la_img_features = global_la_img_features

        elif args.detection ==1 and args.ris == 1:
            if args.mask == 1:
                if args.double_encoder != 1:
                    maksed_good_img_features = model.encode_image(test_good_dataset[:len(test_good_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                    #maksed_sa_img_features = model.encode_image(test_sa_dataset[:len(test_sa_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                    maksed_la_img_features = model.encode_image(test_la_dataset[:len(test_la_dataset)][1][:,rule_idx,:,:,:].squeeze(1))

                else:
                    maksed_good_img_features = model.encode_mask_image(test_good_dataset[:len(test_good_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                    #maksed_sa_img_features = model.encode_image(test_sa_dataset[:len(test_sa_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                    maksed_la_img_features = model.encode_mask_image(test_la_dataset[:len(test_la_dataset)][1][:,rule_idx,:,:,:].squeeze(1))
                good_img_features = ((1.0 - args.beta_mask) * global_good_img_features + args.beta_mask * maksed_good_img_features)
                #sa_img_features = (sa_img_features + maksed_sa_img_features)/2
                la_img_features = ((1.0 - args.beta_mask) * global_la_img_features + args.beta_mask * maksed_la_img_features)
            else:
                good_img_features = global_good_img_features
                la_img_features = global_la_img_features
            

        good_img_features /= good_img_features.norm(dim=-1, keepdim=True)
        #sa_img_features /= sa_img_features.norm(dim=-1, keepdim=True)
        la_img_features /= la_img_features.norm(dim=-1, keepdim=True)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        if args.pos_only == 1:
            # No-negative ablation: normality score = similarity to the positive rule
            # only (no negatives in the denominator). Higher cosine = more normal.
            good_probs.append((good_img_features @ text_features[0:1].T)[:, 0])
            la_probs.append((la_img_features @ text_features[0:1].T)[:, 0])
        else:
            text_features = text_features[:1+args.num_negative]

            #Sample에 대한 good일 확률 계산. positive rule에 대한 similarity is considered as good sample probablity
            good_text_probs = (100.0 * good_img_features @ text_features.T).softmax(dim=-1)
            good_probs.append(good_text_probs[:,0])

            #sa_text_probs = (100.0 * sa_img_features @ text_features.T).softmax(dim=-1)
            #sa_probs.append(sa_text_probs[:,0])

            la_text_probs = (100.0 * la_img_features @ text_features.T).softmax(dim=-1)
            la_probs.append(la_text_probs[:,0])

    test_good_paths = _dataset_img_paths(test_good_dataset)
    test_la_paths = _dataset_img_paths(test_la_dataset)

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
    #set_trace()
    #sa_img_pred = torch.cat((good_probs, sa_probs)).detach().cpu().numpy()
    la_img_pred = torch.cat((good_probs, la_probs)).detach().cpu().numpy()
    if hybrid_score_bank is not None:
        hybrid_good_prob = np.array([hybrid_score_bank.get(path, 0.5) for path in test_good_paths], dtype=np.float32)
        hybrid_la_prob = np.array([hybrid_score_bank.get(path, 0.5) for path in test_la_paths], dtype=np.float32)
        la_img_hybrid_pred = np.concatenate((hybrid_good_prob, hybrid_la_prob))
        la_img_pred = ((1.0 - args.hybrid_weight) * la_img_pred) + (args.hybrid_weight * la_img_hybrid_pred)
    #sa_labels = [1 for i in range(len(good_probs))] + [0 for i in range(len(sa_probs))]
    #la_labels = [1 for i in range(len(good_probs))] + [0 for i in range(len(la_probs))]
    la_labels = [1 for i in range(len(good_probs))] + [0 for i in range(len(la_probs))]
    #sa_auc = roc_auc_score(y_true=sa_labels, y_score=sa_img_pred)
    la_auc = roc_auc_score(y_true=la_labels, y_score=la_img_pred)
    #set_trace()
    auprc, optimal_f1 = calculate_auprc_and_optimal_f1(true_label=la_labels, pred_score=la_img_pred)
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
    return sa_auc, la_auc, auprc, optimal_f1


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
            if len(img) == 2:
                raw_image, masked_image = img
            elif len(img) == 4:
                raw_image, masked_image, _, _ = img                
            global_image_feature = model.encode_image(raw_image)

            for rule_idx, pair in enumerate(pairs):

                if args.mask == 1:
                    masked_image_for_rule = masked_image[:,rule_idx,:,:,:].squeeze(1)
                    if args.double_encoder != 1:
                        masked_image_feature = model.encode_image(masked_image_for_rule)
                    else:
                        masked_image_feature = model.encode_mask_image(masked_image_for_rule)
                    image_feature = ((1.0 - args.beta_mask) * global_image_feature + args.beta_mask * masked_image_feature)
                else:
                    image_feature = model.encode_image(raw_image)
                
                if args.text == 1:
                    local_text = tokenizer(args.texts_for_rules[rule_idx]).to(device)
                    global_text_feature = model.encode_text(pair)
                    local_text_feature = model.encode_text(local_text)
                    text_feature = ((1.0 - args.beta_text) * global_text_feature + args.beta_text * local_text_feature)
                else:
                    text_feature = model.encode_text(pair)

                
                loss = model.clip_loss(image_feature, text_feature, model.clip.logit_scale.exp())
                total_loss += loss
            loss_sum+=total_loss.detach()

    elif args.detection==1 and args.ris == 1:
        for batch, img in enumerate(valid_dataloader):
            total_loss = 0
            raw_image, masked_image, meta = img
            for rule_idx, pair in enumerate(pairs):

                if args.mask == 1:
                    global_image_feature = model.encode_image(raw_image)
                    masked_image_for_rule = masked_image[:,rule_idx,:,:,:].squeeze(1)

                    if args.double_encoder != 1:
                        masked_image_feature = model.encode_image(masked_image_for_rule)
                    else:
                        masked_image_feature = model.encode_mask_image(masked_image_for_rule)
                    image_feature = ((1.0 - args.beta_mask) * global_image_feature + args.beta_mask * masked_image_feature)
                else:
                    image_feature = model.encode_image(raw_image)
                
                if args.text == 1:
                    local_text = tokenizer(args.texts_for_rules[rule_idx]).to(device)
                    global_text_feature = model.encode_text(pair)
                    local_text_feature = model.encode_text(local_text)
                    text_feature = ((1.0 - args.beta_text) * global_text_feature + args.beta_text * local_text_feature)
                else:
                    text_feature = model.encode_text(pair)

                meta_feature_for_rule = meta[:, rule_idx]
                    
                meta_image_feature = torch.cat((meta_feature_for_rule, image_feature), dim=1)
                meta_image_feature = meta_projecter(meta_image_feature)
                loss = model.clip_loss(meta_image_feature, text_feature, model.clip.logit_scale.exp())
                total_loss += loss
            loss_sum+=total_loss.detach()

    if args.log == 1:
        print(f"Valid Loss: {loss_sum/len(valid_dataloader)}")
    return loss_sum



def train(args, model, tokenizer, pairs, train_dataloader, valid_dataloader, test_good_dataset, test_sa_dataset, test_la_dataset, optimizer, epoch=1, tag="default", hybrid_score_bank=None):
    start = time.time()
    best_la_auc = 0
    best_la_f1 = 0
    best_la_aupc = 0
    prev_la_auc = None
    auc_drop_streak = 0
    early_stopped = False
    epochs_executed = 0

    for i in range(epoch):
        epochs_executed = i + 1
        loss_sum = 0
        neg_loss_sum = 0

        
        if i % 10 == 0:
            pass
            #val_loss = valid(args, model, tokenizer, pairs, valid_dataloader)
        with torch.no_grad():
            sa_auc, la_auc, la_auprc, la_f1_max = evaluate(
                args,
                model,
                tokenizer,
                pairs,
                test_good_dataset,
                test_sa_dataset,
                test_la_dataset,
                hybrid_score_bank=hybrid_score_bank,
            )
        if args.log:
            print(f"[Test ROC-AUC]  -  SA: {sa_auc:.5f},  LA: {la_auc:.5f}")
            print(f"[Test AUPRC]  -  LA: {la_auprc:.5f}")
            print(f"[Test F1-max]  -  LA: {la_f1_max:.5f}")

        if prev_la_auc is not None:
            if la_auc < prev_la_auc:
                auc_drop_streak += 1
            else:
                auc_drop_streak = 0
        prev_la_auc = la_auc

        if args.early_stop_k > 0 and auc_drop_streak >= args.early_stop_k:
            print(
                f"[Early Stopping] LA AUROC dropped for {auc_drop_streak} consecutive epochs. "
                f"Stop at epoch {i+1}/{epoch}."
            )
            early_stopped = True
            break

        if la_auc > best_la_auc:
            #if la_auc > 0.9 and (la_auc-best_la_auc) >= 0.01:
                #saved_path = save_model(args, model=model, performance_metric=la_auc, epoch=i, tag=tag, base_directory="./pretrained")
                #pass
            best_la_auc = la_auc
        if la_auprc > best_la_aupc:
            best_la_aupc = la_auprc

        if la_f1_max > best_la_f1:
            best_la_f1 = la_f1_max

        
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

                    # meta_optimizer.step()
                    # meta_optimizer.zero_grad()
                else:
                    pass

                loss_sum+=total_loss.detach()
            if args.log==1:
                print(f"Epoch: {i+1} | Loss: {loss_sum/len(train_dataloader)}")


        elif args.detection ==1 and args.ris != 1:
            for idx, img in enumerate(train_dataloader):
                
                if len(img) == 2:
                    raw_image, masked_image = img
                elif len(img) == 4:
                    raw_image, masked_image, _, _ = img

                total_loss = 0
                total_neg_loss = 0
                global_image_feature = model.encode_image(raw_image)
                
                for rule_idx, pair in enumerate(pairs):
                    if args.rule_select != "":
                        if rule_idx +1 not in args.rule_list:
                            continue

                    if args.mask==1:
                        masked_image_for_rule = masked_image[:,rule_idx,:,:,:].squeeze(1)

                        if args.double_encoder != 1:
                            masked_image_feature = model.encode_image(masked_image_for_rule)
                        else:
                            masked_image_feature = model.encode_mask_image(masked_image_for_rule)
                        image_feature = ((1.0 - args.beta_mask) * global_image_feature + args.beta_mask * masked_image_feature)
                    else:
                        image_feature = global_image_feature


                    if args.text ==1:
                        local_text = tokenizer(args.texts_for_rules[rule_idx]).to(device)
                        global_text_feature = model.encode_text(pair)
                        local_text_feature = model.encode_text(local_text)
                        text_feature = ((1.0 - args.beta_text) * global_text_feature + args.beta_text * local_text_feature)
                    else:
                        text_feature = model.encode_text(pair)

                    if args.pos_only == 1:
                        # No-negative ablation: align images with the positive rule only
                        # (no negative rules in the contrast), to test whether the
                        # positive-vs-negative formulation is necessary.
                        img_n = image_feature / image_feature.norm(dim=-1, keepdim=True)
                        pos_n = text_feature[0:1] / text_feature[0:1].norm(dim=-1, keepdim=True)
                        loss = (1.0 - (img_n @ pos_n.T)).mean()
                    else:
                        loss = model.clip_loss(image_feature, text_feature, model.clip.logit_scale.exp())

                    negative_loss = 0
                    if args.neg_loss != 0:
                        num_img = image_feature.shape[0]
                        if num_img == 1:            #0,| 1, 2, 3, 4, 5
                            negative_loss = compute_negative_count_loss(image_feature, text_feature[0], text_feature[1:1+args.num_negative])
                        else:
                            for num_img_idx in range(num_img):
                                negative_loss += compute_negative_count_loss(image_feature[num_img_idx], text_feature[0], text_feature[1:1+args.num_negative])
                            negative_loss /= num_img
                        
                    else:
                        negative_loss = 0
                        

                    total_loss += (loss + args.neg_loss * negative_loss)

                    total_neg_loss += negative_loss

                if not args.default:
                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # meta_optimizer.step()
                    # meta_optimizer.zero_grad()
                else:
                    pass

                loss_sum+=total_loss.detach().cpu().numpy()

                if args.neg_loss != 0:
                    neg_loss_sum += total_neg_loss.detach().cpu().numpy()
                else:
                    neg_loss_sum += 0
            
            if args.log == 1:
                print(f"Epoch: {i} | Loss: {loss_sum/len(train_dataloader)}  | Negative Loss: {neg_loss_sum/len(train_dataloader)}")
                if loss_sum/len(train_dataloader) == 0:
                    break

        elif args.detection ==1 and args.ris == 1:
            print("Meta feature")
            #meta_projecter = meta_projecter.to(device)
            
            for idx, img in enumerate(train_dataloader):
                raw_image, masked_image, meta = img
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
                        image_feature = ((1.0 - args.beta_mask) * global_image_feature + args.beta_mask * masked_image_feature)
                    else:
                        
                        image_feature = global_image_feature


                    if args.text ==1:
                        local_text = tokenizer(args.texts_for_rules[rule_idx]).to(device)
                        global_text_feature = model.encode_text(pair)
                        local_text_feature = model.encode_text(local_text)
                        text_feature = ((1.0 - args.beta_text) * global_text_feature + args.beta_text * local_text_feature)
                    else:
                        text_feature = model.encode_text(pair)

                    meta_feature_for_rule = meta[:, rule_idx]
                    
                    meta_image_feature = torch.cat((meta_feature_for_rule, image_feature), dim=1)
                    meta_image_feature = meta_projecter(meta_image_feature)
                    
                    loss = model.clip_loss(meta_image_feature, text_feature, model.clip.logit_scale.exp())
                    
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
        if i != 0 and loss_sum/len(train_dataloader) == 0:
            break
        
    end = time.time()

    print(f"Time consumption: {end-start}")
    print(f"Best LA AUC: {best_la_auc}")
    print(f"Best LA AUPC: {best_la_aupc}")
    print(f"Best LA F1-max: {best_la_f1}")
    return {
        "best_la_auc": float(best_la_auc),
        "best_la_aupc": float(best_la_aupc),
        "best_la_f1": float(best_la_f1),
        "epochs_target": int(epoch),
        "epochs_executed": int(epochs_executed),
        "early_stop_k": int(args.early_stop_k),
        "early_stopped": bool(early_stopped),
        "train_seconds": float(end - start),
    }

import open_clip

def main(args):
    
    object_names, objects_for_rules, texts_for_rules = get_object_info(args.data_name, ris_info=False)
    args.object_names = object_names
    args.objects_for_rules = objects_for_rules
    args.texts_for_rules = texts_for_rules
    #print(args)
    #print("Loading model")
    my_model, preprocess, tokenizer, optimizer = get_model(args, model_name=args.model_name, pretrained_name=args.pretrained_name, default=args.default, double_encoder=args.double_encoder)
    
    #preprocess.transforms = preprocess.transforms[:4]
    #set_trace()
    #print("Tokenizing rules")
    if args.num_rule == 0:
        rules_for_data = get_rule(data_name = args.data_name, rule_idxs="max")
    else:
        rules_for_data = get_rule(data_name = args.data_name, rule_idxs=[i for i in range(args.num_rule)])

    # Hand-authored ("manual") negative rules per category (R3-1 generation-strategy ablation).
    # Each entry mirrors the positive-rule list and provides plausible, human-written violations.
    _MANUAL_NEG_RULES = {
        "breakfast": [
            ["There are not two mandarins on the left.", "One of the two mandarins on the left is missing.", "There are three or more mandarins on the left.", "A mandarin on the left is replaced by another fruit.", "The mandarins are placed on the right instead of the left."],
            ["There is no peach on the left.", "The peach is missing from the box.", "The peach is replaced by a different fruit.", "An extra peach is present.", "The peach is placed on the wrong side."],
            ["There is no oat cereal on the top right.", "The oat cereal compartment is empty.", "The oat cereal is replaced by a different cereal.", "The oat cereal is placed in the wrong compartment.", "There is far too little oat cereal on the top right."],
            ["There are no almonds down the right.", "The almonds are missing.", "The almonds are replaced by another nut.", "The almonds are placed in the wrong compartment.", "There are far too few almonds down the right."],
            ["There are no banana chips down the right.", "The banana chips are missing.", "The banana chips are replaced by another snack.", "The banana chips are placed in the wrong compartment.", "There are far too few banana chips down the right."],
            ["Some content overflows its compartment.", "An item spills over into a neighboring compartment.", "The contents exceed their designated boundaries.", "Food is piled above the rim of a compartment.", "Items are scattered outside their compartments."],
            ["There are far more banana chips than almonds.", "There are far more almonds than banana chips.", "The amounts of banana chips and almonds are clearly unequal.", "One of the two snack compartments is nearly empty.", "Banana chips and almonds are present in very different quantities."],
        ],
        "splicing": [
            ["There are not two splicing connectors.", "One splicing connector is missing.", "There are three or more splicing connectors.", "Only a single splicing connector is present.", "The number of splicing connectors is wrong."],
            ["The two connectors are at clearly different heights.", "One connector is much higher than the other.", "The connectors are misaligned vertically.", "The two connectors are not level with each other.", "There is a large height gap between the two connectors."],
            ["More than one cable is connected.", "No cable is connected.", "Two or more cables are attached.", "The number of connected cables is wrong.", "Several cables are plugged in at once."],
            ["The two-block connector has a cable of the wrong color.", "The two-block connector is connected with a blue cable.", "The two-block connector is connected with an orange cable.", "The two-block connector's cable is not yellow.", "A non-yellow cable is attached to the two-block connector."],
            ["The three-block connector has a cable of the wrong color.", "The three-block connector is connected with a yellow cable.", "The three-block connector is connected with an orange cable.", "The three-block connector's cable is not blue.", "A non-blue cable is attached to the three-block connector."],
            ["The five-block connector has a cable of the wrong color.", "The five-block connector is connected with a blue cable.", "The five-block connector is connected with a yellow cable.", "The five-block connector's cable is not orange.", "A non-orange cable is attached to the five-block connector."],
        ],
        "banana_juice": [
            ["The bottle has no banana picture label.", "The picture label shows a different fruit, not a banana.", "The banana picture label is missing.", "The label depicts the wrong fruit.", "There is no fruit picture on the label."],
            ["The juice is not white.", "The bottle contains juice of the wrong color.", "The juice color does not match banana juice.", "The liquid is colored differently from banana juice.", "The juice has an incorrect color."],
            ["The picture label is not centered.", "The picture label is shifted off-center.", "The label is misaligned, not in the center.", "The picture label is placed to one side.", "The label position is off from the center."],
            ["The text label is not at the bottom.", "The text label is placed in the wrong position.", "The text label appears at the top instead of the bottom.", "The text label is misplaced.", "The bottom does not contain the text label."],
            ["The bottle is not full of banana juice.", "The bottle is partially empty.", "The juice level is too low.", "The bottle is underfilled.", "There is not enough juice in the bottle."],
        ],
        "orange_juice": [
            ["The bottle has no orange picture label.", "The picture label shows a different fruit, not an orange.", "The orange picture label is missing.", "The label depicts the wrong fruit.", "There is no fruit picture on the label."],
            ["The juice is not yellow.", "The bottle contains juice of the wrong color.", "The juice color does not match orange juice.", "The liquid is colored differently from orange juice.", "The juice has an incorrect color."],
            ["The picture label is not centered.", "The picture label is shifted off-center.", "The label is misaligned, not in the center.", "The picture label is placed to one side.", "The label position is off from the center."],
            ["The text label is not at the bottom.", "The text label is placed in the wrong position.", "The text label appears at the top instead of the bottom.", "The text label is misplaced.", "The bottom does not contain the text label."],
            ["The bottle is not full of orange juice.", "The bottle is partially empty.", "The juice level is too low.", "The bottle is underfilled.", "There is not enough juice in the bottle."],
        ],
        "cherry_juice": [
            ["The bottle has no cherry picture label.", "The picture label shows a different fruit, not a cherry.", "The cherry picture label is missing.", "The label depicts the wrong fruit.", "There is no fruit picture on the label."],
            ["The juice is not red.", "The bottle contains juice of the wrong color.", "The juice color does not match cherry juice.", "The liquid is colored differently from cherry juice.", "The juice has an incorrect color."],
            ["The picture label is not centered.", "The picture label is shifted off-center.", "The label is misaligned, not in the center.", "The picture label is placed to one side.", "The label position is off from the center."],
            ["The text label is not at the bottom.", "The text label is placed in the wrong position.", "The text label appears at the top instead of the bottom.", "The text label is misplaced.", "The bottom does not contain the text label."],
            ["The bottle is not full of cherry juice.", "The bottle is partially empty.", "The juice level is too low.", "The bottle is underfilled.", "There is not enough juice in the bottle."],
        ],
        "pushpins": [
            ["There are not fifteen pushpins.", "The number of pushpins is wrong.", "There are fewer than fifteen pushpins.", "There are more than fifteen pushpins.", "One or more pushpins are missing."],
            ["Some pushpins are not separated by the plastic case.", "Two pushpins share the same compartment.", "The plastic dividers do not separate every pushpin.", "Pushpins are grouped together without separation.", "The case fails to keep each pushpin apart."],
            ["A compartment contains more than one pushpin.", "Some parts hold two or more pushpins.", "More than one pushpin is placed in a single slot.", "A slot is occupied by multiple pushpins.", "The one-pushpin-per-compartment rule is violated."],
            ["There is an empty compartment.", "Some slots are left blank.", "A compartment has no pushpin.", "There are vacant parts without a pushpin.", "One or more slots are empty."],
        ],
        "screw_bag": [
            ["There are not two bolts.", "The number of bolts is wrong.", "One bolt is missing.", "There are three or more bolts.", "Only a single bolt is present."],
            ["There is no short bolt.", "The short bolt is missing.", "Both bolts are long, none is short.", "The short bolt is replaced by a long one.", "A short bolt is absent from the bag."],
            ["There is no long bolt.", "The long bolt is missing.", "Both bolts are short, none is long.", "The long bolt is replaced by a short one.", "A long bolt is absent from the bag."],
            ["There are not two hexagonal nuts.", "The number of hexagonal nuts is wrong.", "One hexagonal nut is missing.", "There are three or more hexagonal nuts.", "A nut of the wrong type replaces a hexagonal nut."],
            ["There are not two round washers.", "The number of round washers is wrong.", "One round washer is missing.", "There are three or more round washers.", "A washer of the wrong type replaces a round washer."],
        ],
    }

    # Positive rule quality ablation (vague / paraphrase variants)
    _VAGUE_POS_RULES = {
        "breakfast": [
            "There must be some citrus fruit items in the image.",
            "There must be some stone fruit items in the image.",
            "There must be some cereal items in the image.",
            "There must be some nut items in the image.",
            "There must be some chip items in the image.",
            "Items should be properly placed in the box.",
            "Items should be properly arranged.",
        ],
        "splicing": [
            "There must be some connectors in the image.",
            "The connectors should look similar in size.",
            "There must be some cables attached.",
            "Some connectors must have some color cables.",
            "Some connectors must have some color cables.",
            "Some connectors must have some color cables.",
        ],
        "juice_bot": [
            "There should be a juice bottle with some fruit label.",
            "The bottle should contain some juice.",
            "There should be a juice bottle with some fruit label.",
            "The bottle should contain some colored juice.",
            "There should be a juice bottle with some fruit label.",
            "The bottle should contain some colored juice.",
            "There should be a label on the bottle.",
            "There should be some text on the bottle.",
        ],
        "pushpins": [
            "There should be many pushpins.",
            "The pushpins should be in compartments.",
            "Each compartment should hold a pushpin.",
            "The compartments should be filled.",
        ],
        "screw_bag": [
            "There should be some bolts.",
            "There should be a smaller bolt.",
            "There should be a larger bolt.",
            "There should be some nuts.",
            "There should be some washers.",
        ],
    }
    _PARAPHRASE_POS_RULES = {
        "breakfast": [
            "Two tangerines must be positioned on the left side.",
            "A peach must be located on the left portion of the image.",
            "Oat cereal must appear in the upper right section.",
            "Almonds need to be placed at the lower right.",
            "Banana chips must be found at the lower right.",
            "None of the contents should overflow their designated area.",
            "Banana chips and almonds should have equal quantities.",
        ],
        "splicing": [
            "The image should contain exactly two splicing connectors.",
            "Both connectors should be at the same vertical height.",
            "Exactly one cable should be in a connected state.",
            "The two-block connector should be equipped with a yellow cable.",
            "The three-block connector should be equipped with a blue cable.",
            "The five-block connector should be equipped with an orange cable.",
        ],
        "juice_bot": [
            "The banana juice bottle should display a banana image on its label.",
            "The juice inside the banana bottle should be white.",
            "The cherry juice bottle should display a cherry image on its label.",
            "The juice inside the cherry bottle should be red.",
            "The orange juice bottle should display an orange image on its label.",
            "The juice inside the orange bottle should be yellow.",
            "The picture on the label should be positioned at the center.",
            "The text on the label should be located at the bottom.",
        ],
        "pushpins": [
            "The case should contain exactly fifteen pushpins.",
            "Each pushpin should be kept in its own plastic compartment.",
            "No compartment should hold more than one pushpin.",
            "There should be no empty compartment.",
        ],
        "screw_bag": [
            "The bag should contain two bolts.",
            "One of the bolts should be the short one.",
            "One of the bolts should be the long one.",
            "There should be a pair of hexagonal nuts.",
            "There should be a pair of round washers.",
        ],
    }
    if args.pos_rule_type != 'original':
        _rule_map = _VAGUE_POS_RULES if args.pos_rule_type == 'vague' else _PARAPHRASE_POS_RULES
        if args.data_name not in _rule_map:
            raise ValueError(f"--pos_rule_type '{args.pos_rule_type}' not defined for dataset '{args.data_name}'")
        new_pos = _rule_map[args.data_name]
        # Align length with loaded rules
        rules_for_data[0] = new_pos[:len(rules_for_data[0])]

    # Negative rule quality ablation (R3-1): compare generation strategies
    if args.neg_rule_type == 'shuffled':
        neg_rules = rules_for_data[1]
        n = len(neg_rules)
        rules_for_data[1] = [neg_rules[(i + 1) % n] for i in range(n)]
    elif args.neg_rule_type == 'naive':
        # Unvalidated LLM-style generic negatives (no semantic checker)
        pos_rules = rules_for_data[0]
        rules_for_data[1] = [
            [
                f"This image does not follow the rule: {pos}",
                "The arrangement shown violates product specifications.",
                "The components are not in the correct configuration.",
                "There is an anomaly visible in the arrangement.",
                "The image shows an incorrect product state.",
            ]
            for pos in pos_rules
        ]
    elif args.neg_rule_type == 'template':
        # Deterministic string templates derived from each positive rule (no LLM)
        pos_rules = rules_for_data[0]
        def _pl(p):
            p = p.strip()
            return (p[0].lower() + p[1:]).rstrip('.')
        rules_for_data[1] = [
            [
                f"This image does not satisfy the rule: {pos.strip().rstrip('.')}.",
                f"The condition is violated: {pos.strip().rstrip('.')}.",
                f"It is not the case that {_pl(pos)}.",
                f"The image fails the requirement that {_pl(pos)}.",
                f"The arrangement does not meet the rule: {pos.strip().rstrip('.')}.",
            ]
            for pos in pos_rules
        ]
    elif args.neg_rule_type == 'qa':
        # QA-style negatives (reproducing the conference-version construction)
        pos_rules = rules_for_data[0]
        def _pl(p):
            p = p.strip()
            return (p[0].lower() + p[1:]).rstrip('.')
        rules_for_data[1] = [
            [
                f"Question: Does the image satisfy '{pos.strip().rstrip('.')}'? Answer: No.",
                f"Q: Is it true that {_pl(pos)}? A: No, it is not.",
                f"Asked whether {_pl(pos)}, the answer is negative.",
                f"Does this image meet the rule '{pos.strip().rstrip('.')}'? It does not.",
                f"Is the rule '{pos.strip().rstrip('.')}' satisfied? No.",
            ]
            for pos in pos_rules
        ]
    elif args.neg_rule_type == 'manual':
        # Hand-authored negatives encoding plausible violations (domain knowledge)
        if args.data_name not in _MANUAL_NEG_RULES:
            raise ValueError(f"--neg_rule_type 'manual' not defined for dataset '{args.data_name}'")
        man = _MANUAL_NEG_RULES[args.data_name]
        if len(man) != len(rules_for_data[1]):
            raise ValueError(
                f"manual neg rules count ({len(man)}) != positive rules count ({len(rules_for_data[1])}) for {args.data_name}")
        rules_for_data[1] = man

    rule_token_pairs = get_rule_tokens(rules=rules_for_data, tokenizer=tokenizer, device=device)
    
    print(args)
    #print("Generating dataloaders")
    data_path_dict = get_data_path(args.data_name, log_option=False)

    if args.data_name not in DATA_BASE_PATH_DICT:
        raise ValueError(
            f"Unsupported data_name: {args.data_name}. "
            f"Supported products: {sorted(DATA_BASE_PATH_DICT.keys())}"
        )

    args.data_base_path = DATA_BASE_PATH_DICT[args.data_name]
    # Non-destructive override: load precomputed .pt from a custom directory
    # (e.g. an alternate kernel-size RRD build) without touching the main data.
    if getattr(args, 'data_base_path_override', '') :
        args.data_base_path = args.data_base_path_override
        print(f"[data_base_path override] loading .pt from {args.data_base_path}")


    #train_dataset = torch.load(os.path.join(args.data_base_path, f"{tag}_train.pt"))
    #val_dataset = torch.load(os.path.join(args.data_base_path, f'{tag}_val.pt'))
    #test_good_dataset = torch.load(os.path.join(args.data_base_path, f'{tag}_test_good.pt'))
    #test_la_dataset = torch.load(os.path.join(args.data_base_path, f'{tag}_test_la.pt'))

    train_dataset = torch.load(os.path.join(args.data_base_path, f"train.pt"), weights_only=False)
    test_good_dataset = torch.load(os.path.join(args.data_base_path, f'test_good.pt'), weights_only=False)
    test_la_dataset = torch.load(os.path.join(args.data_base_path, f'test_la.pt'), weights_only=False)
    
    # train_dataset = ad_dataset(data_path_dict, 'train_path', preprocess=preprocess, level=3, device=device)
    # test_good_dataset = ad_dataset(data_path_dict, 'test_good_path', preprocess=preprocess, level=3, device=device)
    # test_la_dataset = ad_dataset(data_path_dict, 'test_la_path', preprocess=preprocess, level=3, device=device)
    test_sa_dataset = None

    if args.few_shot == 1:
        sampled_indices = random.sample(range(len(train_dataset)), args.shot)
        train_dataset = Subset(train_dataset, sampled_indices)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=(len(train_dataset) >= args.batch_size),
    )
    val_dataloader = None

    hybrid_score_bank = build_hybrid_score_bank(
        args=args,
        train_dataset=train_dataset,
        test_good_dataset=test_good_dataset,
        test_la_dataset=test_la_dataset,
        device=device,
    )
    hybrid_only_result = evaluate_hybrid_only(
        test_good_dataset=test_good_dataset,
        test_la_dataset=test_la_dataset,
        hybrid_score_bank=hybrid_score_bank,
    )
    if hybrid_only_result is not None and args.log:
        print(
            "[Hybrid-only] "
            f"LA_AUC={hybrid_only_result['hybrid_only_la_auc']:.5f}, "
            f"LA_AUPRC={hybrid_only_result['hybrid_only_la_auprc']:.5f}, "
            f"LA_F1={hybrid_only_result['hybrid_only_la_f1']:.5f}"
        )
    #set_trace()
    
    print(f"[Data Description]")
    print(f"- Train: {len(train_dataset)}")
    #print(f"- Valid: {len(val_dataset)}")
    print(f"- Test: {len(test_good_dataset)+len(test_la_dataset)}\n")
    
    my_model = my_model.to(device)
    print("Train Start")
    if not args.ciriculam_learning:
        train_result = train(
            args,
            model=my_model,
            tokenizer=tokenizer,
            pairs=rule_token_pairs,
            train_dataloader=train_dataloader,
            valid_dataloader=val_dataloader,
            test_good_dataset=test_good_dataset,
            test_sa_dataset=test_sa_dataset,
            test_la_dataset=test_la_dataset,
            optimizer=optimizer,
            epoch=args.epoch,
            tag=args.tag,
            hybrid_score_bank=hybrid_score_bank,
        )
        append_experiment_result(args, train_result, hybrid_only_result=hybrid_only_result)
    else:
        for num_rule in range(1, len(rule_token_pairs)):
            print(f"=== [{num_rule-1} ciriculam] ===")
            temp_rule_token_pairs = rule_token_pairs[:num_rule]
            train_result = train(
                args,
                model=my_model,
                tokenizer=tokenizer,
                pairs=temp_rule_token_pairs,
                train_dataloader=train_dataloader,
                valid_dataloader=val_dataloader,
                test_good_dataset=test_good_dataset,
                test_sa_dataset=test_sa_dataset,
                test_la_dataset=test_la_dataset,
                optimizer=optimizer,
                epoch=args.epoch,
                tag=args.tag,
                hybrid_score_bank=hybrid_score_bank,
            )
            append_experiment_result(args, train_result, hybrid_only_result=hybrid_only_result)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--gpu_number', default="0", type=str)
    parser.add_argument('--data_name', default='breakfast', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
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
    parser.add_argument('--rule_select', default="", type=str)
    parser.add_argument('--rule_list', default=[], type=list)
    parser.add_argument('--num_negative', default=5, type=int)
    parser.add_argument('--hybrid_model', default="none", type=str, choices=["none", "winclip", "promptad"])
    parser.add_argument('--hybrid_weight', default=0.1, type=float)
    parser.add_argument('--hybrid_class_name', default="", type=str)
    parser.add_argument('--hybrid_batch_size', default=400, type=int)
    parser.add_argument('--hybrid_backbone', default='ViT-B-16-plus-240', type=str)
    parser.add_argument('--hybrid_pretrained_dataset', default='laion400m_e32', type=str)
    parser.add_argument('--hybrid_scales', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--hybrid_img_resize', default=240, type=int)
    parser.add_argument('--hybrid_img_cropsize', default=240, type=int)
    parser.add_argument('--hybrid_resolution', default=400, type=int)
    parser.add_argument('--hybrid_promptad_epoch', default=100, type=int)
    parser.add_argument('--hybrid_promptad_lr', default=0.002, type=float)
    parser.add_argument('--hybrid_promptad_momentum', default=0.9, type=float)
    parser.add_argument('--hybrid_promptad_weight_decay', default=5e-4, type=float)
    parser.add_argument('--hybrid_promptad_eta_min', default=1e-5, type=float)
    parser.add_argument('--hybrid_promptad_lambda1', default=0.001, type=float)
    parser.add_argument('--hybrid_promptad_n_ctx', default=4, type=int)
    parser.add_argument('--hybrid_promptad_n_pro', default=3, type=int)
    parser.add_argument('--hybrid_promptad_n_ctx_ab', default=1, type=int)
    parser.add_argument('--hybrid_promptad_n_pro_ab', default=4, type=int)
    parser.add_argument('--early_stop_k', default=10, type=int)
    parser.add_argument('--beta_mask', default=0.5, type=float)   # weight for masked vs global image feature (0=global only, 1=masked only)
    parser.add_argument('--beta_text', default=0.5, type=float)   # weight for local vs global text feature (0=global only, 1=local only)
    parser.add_argument('--pos_rule_type', default='original', type=str, choices=['original', 'vague', 'paraphrase'])
    parser.add_argument('--neg_rule_type', default='original', type=str, choices=['original', 'shuffled', 'naive', 'template', 'qa', 'manual'])
    parser.add_argument('--result_log_path', default='./results/train_hybrid_result.csv', type=str)
    parser.add_argument('--data_base_path_override', default='', type=str, help='load precomputed train/test .pt from this dir instead of the default (non-destructive, e.g. alternate kernel RRD build)')
    parser.add_argument('--pos_only', default=0, type=int, help='ablation: train and score with positive rules only (no negative rules anywhere) to test the necessity of negatives')
    args = parser.parse_args()
    #--model_name ViT-g-14 --pretrained_name laion2b_s12b_b42k

    if args.rule_select != "":
        selected_rule = args.rule_select.split("_")
        selected_rule = [int(rule) for rule in selected_rule]
        selected_rule.sort()
        args.rule_list = selected_rule

    if args.hybrid_batch_size < 1:
        raise ValueError("--hybrid_batch_size must be >= 1")
    if args.hybrid_promptad_epoch < 0:
        raise ValueError("--hybrid_promptad_epoch must be >= 0")
    if args.hybrid_promptad_eta_min < 0:
        raise ValueError("--hybrid_promptad_eta_min must be >= 0")
    if not (0.0 <= args.hybrid_weight <= 1.0):
        raise ValueError("--hybrid_weight must be in [0, 1]")
    if args.early_stop_k < 0:
        raise ValueError("--early_stop_k must be >= 0")
    #model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    if args.seed != 0:
        set_random_seed(random_seed=args.seed)
    else:
        pass

    print(f"Torch version: {torch.__version__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is loaded for process")
    
    meta_projecter = meta_projecter.to(device)
    #args.log=False
    main(args)
    
