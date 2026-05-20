import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
warnings.simplefilter(action='ignore', category=UserWarning)
import os
import sys
sys.path.append(os.path.abspath("/home/seungeon/Workspace/"))
sys.path.append(os.path.abspath("/home/seungeon/Workspace/segment-anything"))
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
                                
sys.path.append(os.path.abspath("/home/seungeon/Workspace/zero-shot-RIS/third_party/modified_CLIP"))
sys.path.append(os.path.abspath("/home/seungeon/Workspace/zero-shot-RIS/third_party/FreeSOLO"))
sys.path.append(os.path.abspath("/home/seungeon/Workspace/zero-shot-RIS/third_party/old_detectron2"))
import clip
from clip.simple_tokenizer import SimpleTokenizer
import argparse

sys.path.append(os.path.abspath("/home/seungeon/Workspace/zero-shot-RIS"))
from utils import default_argument_parser, setup

# FreeSOLO
from detectron2.checkpoint import DetectionCheckpointer
from freesolo.engine.trainer import BaselineTrainer
import freesolo.data.datasets.builtin
from freesolo.modeling.solov2 import PseudoSOLOv2
from ad_clip.utils_save import get_rule, get_data_path, get_rule_tokens, get_object_info
from ad_clip.data import ad_lisa_dataset
import spacy
from pdb import set_trace

from random import shuffle

def freesolo_preprocess(image):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # normalize를 위한 평균, 표준편차 값 지정
    width, height = image.size # 이미지의 넓이와 높이 정보 저장

    transform = T.Compose([T.Resize(800), # Image 크기 조정
                        T.ToTensor(), # Tensor로 변환
                        T.Normalize(mean, std) # Normalize
                        ])

    resized_img = transform(image).unsqueeze(0)

    input_img = dict(image=resized_img, height=height, width=width) # dictionary 형태로 변경
    return input_img

clip_transform = T.Resize((224,224))
def clip_preprocess(img, device='cuda'):
    input_img = clip_transform(img.to(device))
    return input_img


def main(args):
    cfg = default_argument_parser().parse_args(args=[])
    cfg = setup(cfg)
    device = 'cuda'
    Trainer = BaselineTrainer
    Free_SOLO = Trainer.build_model(cfg) # detectron의 Trainer Class를 사용하여 모델 불러오기
    Free_SOLO = Free_SOLO.eval() # evaluation 모드로 전환

    path2weights = '/home/seungeon/Workspace/zero-shot-RIS/checkpoints/FreeSOLO_R101_30k_pl.pth' # pre-trained weights 경로 지정
    _ = DetectionCheckpointer(Free_SOLO).resume_or_load(path2weights, resume=True) # pre-trained weights 불러오기

    
    clip_model, _ = clip.load('RN50')

    nlp = spacy.load('en_core_web_lg') # dependency parsing tool

    #Data setting
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
    object_names, objects_for_rules, lisa_prompt_targets = get_object_info(args.data_name, ris_info=False)

    base_save_path = data_base_path_dict[args.data_name]

    data_path_dict = get_data_path(args.data_name)

    path_options = ['train_path', 'val_path', 'test_good_path', 'test_la_path']
    shuffle(path_options)
    for path_option in path_options:

        image_list = []
        lisa_masked_image_list = [[] for _ in range(len(data_path_dict[path_option]))]


        for img_idx, train_img_path in enumerate(data_path_dict[path_option]):
            if not os.path.exists(train_img_path):
                print("File not found in {}".format(train_img_path))
                continue

            
            image_np = cv2.imread(train_img_path)
            image_pil = Image.open(train_img_path).convert("RGB")
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            width, height = image_pil.size

            image_list.append(image_pil)

            input_image = freesolo_preprocess(image_pil)

            pred = Free_SOLO([input_image])[0] # FreeSOLO inference
            


            pred_masks = pred['instances'].pred_masks # output mask
            pred_boxes = pred['instances'].pred_boxes # output boxes

            if len(pred_masks) == 0:
                try:
                    #pred = past_prediction
                    num = 10

                    pred_masks = torch.ones(size=(num, height, width), device="cuda")
                    pred_boxes = torch.stack([torch.tensor([0,0, height, width], device="cuda") for _ in range(num)])
                except:
                    num = 10

                    pred_masks = torch.ones(size=(num, height, width), device="cuda")
                    pred_boxes = torch.stack([torch.tensor([0,0, height, width], device="cuda") for _ in range(num)])
        
            #past_prediction = pred
            print(pred_masks.shape)
            #set_trace()
            print(dir(input_image))
            print(input_image.keys())
            image_tensor = input_image['image']

            clip_input_img = clip_preprocess(image_tensor)
            feature_map = clip_model.encode_image(clip_input_img)
            feature_map = feature_map / feature_map.norm(dim=1, keepdim=True) # normalize feature map

            #set_trace()
            masks = TF.resize(pred_masks.type(torch.float32), (feature_map.shape[2:])) # masks 들을 feature map 크기로 resize 합니다.
             
            masked_feature_map = torch.mul(feature_map, masks[:,None,:,:]) # feature map과 masks 들을 element-wise product 합니다.
            print(masked_feature_map.shape)
            global_visual_features = clip_model.visual.attnpool(masked_feature_map)
            print(global_visual_features.shape)


            pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device) # masked 영역을 mean으로 채웁니다.

            original_img = T.Resize((height, width))(clip_input_img).to(device) # 이미지를 원본 크기로 변환

            cropped_imgs = []

            for m, b in zip(pred_masks, pred_boxes):
                m, b = m.type(torch.uint8), b.type(torch.int) # type 변환
                masked_img = original_img * m[None, None, ...] + (1 - m[None, None, ...]) * pixel_mean

                x_min, y_min, x_max, y_max = b
                h, w = y_max - y_min, x_max - x_min

                cropped_img = TF.resized_crop(masked_img.squeeze(0), y_min, x_min, h, w, (224, 224))
                cropped_imgs.append(cropped_img)

            cropped_imgs = torch.stack(cropped_imgs, dim=0)
            print(cropped_imgs.shape)

            feature_map = clip_model.encode_image(cropped_imgs)
            local_visual_features = clip_model.visual.attnpool(feature_map)

            local_visual_features = local_visual_features / local_visual_features.norm(dim=1, keepdim=True) # normalize

            print(local_visual_features.shape)


            global_local_visual_features = 0.85 * global_visual_features + (1 - 0.85) * local_visual_features
            print(global_local_visual_features.shape)

            for lisa_prompt in lisa_prompt_targets:
                sentence = lisa_prompt[0]

                sentence_token = clip.tokenize(sentence).to(device) # Tokenize

                sentence_feature = clip_model.encode_text(sentence_token)
                sentence_feature = sentence_feature / sentence_feature.norm(dim=1, keepdim=True)

                print(sentence_feature.shape)

                doc = nlp(sentence)
                print('sentence: ', sentence)

                # 문장에서 noun phrase 찾기
                chunks = {}
                for chunk in doc.noun_chunks: # dependency parsing으로 찾은 noun phrase를 하나하나 꺼내오기
                    for i in range(chunk.start, chunk.end): # 문장에서 각 word가 어느 noun phrase에 속하지 확인 
                        chunks[i] = chunk

                print('noun phrase 모음: ',chunks)


                # root word 찾기
                for token in doc:
                    if token.head.i == token.i:
                        root_word = token.head
                        
                print('root word: ', root_word)


                # root word가 속해있는 noun phrase 추출
                try:
                    noun_phrase = chunks[root_word.i].text
                except:
                    noun_phrase = "object"
                print('root word가 포함되어 있는 noun phrase: ',noun_phrase)

                noun_phrase_token = clip.tokenize(noun_phrase).to(device) # Tokenize

                noun_phrase_feature = clip_model.encode_text(noun_phrase_token)
                noun_phrase_feature = noun_phrase_feature / noun_phrase_feature.norm(dim=1, keepdim=True) # normalize

                print(noun_phrase_feature.shape)

                global_local_textual_feature = 0.5 * sentence_feature + (1 - 0.5) * noun_phrase_feature
                print(global_local_textual_feature.shape)

                similarity = global_local_visual_features @ global_local_textual_feature.T

                max_index = torch.argmax(similarity)

                mask_prediction = pred_masks[max_index]
                
                mask = np.logical_not(mask_prediction.detach().cpu().numpy())
                output = image_np.copy()
                output[mask] = 0
                #output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                masked_img = Image.fromarray(output)
                lisa_masked_image_list[img_idx].append(masked_img)

        data_save_path = f"./clip_detection/{args.data_name}/{path_option}/"
        os.makedirs(data_save_path, exist_ok=True)
        train_dataset=ad_lisa_dataset(img_path_list=data_path_dict[path_option], image_list=image_list, 
                                lisa_masked_image_list=lisa_masked_image_list, target_objects_list=objects_for_rules, 
                                save=True, save_path=data_save_path, device="cuda")
        data_option = path_option.replace("_path", "")
        torch.save(train_dataset, os.path.join('./clip_detection',args.data_name, f'clip_{data_option}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--data_name", default="breakfast", type=str)
    args = parser.parse_args()

    main(args)
