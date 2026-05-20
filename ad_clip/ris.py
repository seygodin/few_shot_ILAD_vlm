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
#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
                                
# sys.path.append(os.path.abspath("/home/seungeon/Workspace/zero-shot-RIS/third_party/modified_CLIP"))
# import clip
# from clip.simple_tokenizer import SimpleTokenizer
# import spacy

sys.path.append(os.path.abspath("/home/seungeon/Workspace/"))
import open_clip
import pickle
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # normalize를 위한 평균, 표준편차 값 지정

transform = T.Compose([T.Resize(800), # Image 크기 조정
                      T.ToTensor(), # Tensor로 변환
                      T.Normalize(mean, std) # Normalize
                      ])
def preprocess(image, device="cuda"):
    width, height = image.size # 이미지의 넓이와 높이 정보 저장
    resized_img = transform(image).unsqueeze(0)
    resized_img = resized_img.to(device)
    return resized_img, width, height 
def torch_to_cv2(tensor):
    if tensor.max() <= 1:
        tensor = tensor.mul(255).byte()  # 정규화 해제 및 uint8로 변환
    if len(tensor.shape) > 3:
        tensor = tensor.squeeze(0)
    np_image = tensor.cpu().numpy()  # GPU에서 CPU로 이동
    np_image = np.transpose(np_image, (1, 2, 0))  # C, H, W -> H, W, C

    # OpenCV는 BGR을 사용하므로 RGB를 BGR로 변환
    cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    
    # OpenCV 이미지를 uint8로 변환
    cv_image = cv_image.astype(np.uint8)
    
    # uint8 이미지를 PIL 이미지로 변환
    #pil_image = Image.fromarray(cv_image)

    return cv_image
def PIL2OpenCV(pil_image):
    numpy_image= np.array(pil_image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image
def convert_sam_bbox_to_pil_bbox(bbox):
    x, y, h, w = bbox
    return (x, y, x+w, y+h)
def run_sam(mask_generator, np_image, num_mask, device="cuda"):
    image = np_image
    sam_masks = mask_generator.generate(image)
    sam_masks = sorted(sam_masks, key=lambda x: x["area"], reverse=True)
    tensor_sam_masks = torch.Tensor(np.stack([sam_masks[idx]["segmentation"] for idx in range(len(sam_masks))]))
    tensor_sam_masks = torch.Tensor.bool(tensor_sam_masks)
    tensor_sam_masks = tensor_sam_masks.to(device)
    tensor_sam_boxes = torch.Tensor([convert_sam_bbox_to_pil_bbox(sam_masks[idx]["bbox"]) for idx in range(len(sam_masks))])
    tensor_sam_boxes = tensor_sam_boxes.to(device)
    return tensor_sam_masks, tensor_sam_boxes, sam_masks
input_transform = T.Resize((224,224))

def find_unique_strings(double_list):
    unique_strings = set()  # 고유한 문자열을 저장할 세트 생성
    for sublist in double_list:
        for item in sublist:
            unique_strings.add(item)  # 각 문자열을 세트에 추가하여 고유하게 유지
    return list(unique_strings)  # 세트를 리스트로 변환하여 반환

def extract_global_local_visual_feature(input_img, clip_model, sam_masks, sam_boxes, width, height, local_factor=0.9, crop_image_return=False, device='cuda'):
    resized_img = input_img
    #print(input_img.shape)
    sam_masks = sam_masks.to(device)
    sam_boxes = sam_boxes.to(device)
    input_img = input_transform(input_img)
    
    feature_map = clip_model.encode_image(input_img)
    feature_map = feature_map / feature_map.norm(dim=1, keepdim=True) # normalize feature map
    
    masks = TF.resize(sam_masks.type(torch.float32), (feature_map.shape[2:]), interpolation=Image.NEAREST) # masks 들을 feature map 크기로 resize 합니다.
    masks = masks.to(device)
    masked_feature_map = torch.mul(feature_map, masks[:,None,:,:]) # feature map과 masks 들을 element-wise product 합니다.
    
    global_visual_features = clip_model.visual.attnpool(masked_feature_map)
    
    pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device) # masked 영역을 mean으로 채웁니다.

    original_img = T.Resize((height, width))(resized_img).to(device) # 이미지를 원본 크기로 변환
    #print(original_img.shape)
    cropped_imgs = []

    for m, b in zip(sam_masks, sam_boxes):
        m, b = m.type(torch.uint8), b.type(torch.int) # type 변환
        masked_img = original_img * m[None, None, ...] + (1 - m[None, None, ...]) * pixel_mean

        x_min, y_min, x_max, y_max = b
        h, w = y_max - y_min, x_max - x_min

        if x_min <= 0:
            x_min = 1
        if y_min <= 0:
            y_min = 1
        if h <= 0:
            h = 1
        if w <= 0:
            w = 1

        try: 
            cropped_img = TF.resized_crop(masked_img.squeeze(0), y_min, x_min, h, w, (224, 224))
        except:
            print(y_min, x_min, h, w)
            cropped_img = T.Resize(224,224)(masked_img.squeeze(0))

        cropped_imgs.append(cropped_img)
        
    tensor_cropped_imgs = torch.stack(cropped_imgs, dim=0)
    
    feature_map = clip_model.encode_image(tensor_cropped_imgs)
    local_visual_features = clip_model.visual.attnpool(feature_map)
    local_visual_features = local_visual_features / local_visual_features.norm(dim=1, keepdim=True) # normalize
    
    global_local_visual_features = (1-local_factor) * global_visual_features + local_factor * local_visual_features
    
    if crop_image_return:
        return global_local_visual_features, cropped_imgs
    else:
        return global_local_visual_features


def extract_global_local_text_feature(clip_model, sentence, device='cuda'):
    sentence_token = clip.tokenize(sentence).to(device) # Tokenize

    sentence_feature = clip_model.encode_text(sentence_token)
    sentence_feature = sentence_feature / sentence_feature.norm(dim=1, keepdim=True)
    return sentence_feature

def extract_black_mask(pil_image):
    # RGB 이미지에서 모든 채널이 0인 픽셀을 검은색으로 간주하고 마스크 생성
    # np.all을 사용하여 축(axis=-1)을 기준으로 모든 값이 0인 경우 True 반환
    pil_image = np.array(pil_image)
    mask = np.all(pil_image == 0, axis=-1)
    mask = mask.astype(int) 
    """
    one_mask = np.where(mask == 1)
    zero_mask = np.where(mask ==0)
    mask[one_mask] = 0
    mask[zero_mask] =1
    """
    return mask

def filter_sam_mask(black_mask, tensor_sam_masks, tensor_sam_boxes):
    mask_result = []
    box_result = []
    for sam_mask_idx in range(len(tensor_sam_masks)):
        sam_mask = tensor_sam_masks[sam_mask_idx].detach().cpu().numpy()
        if not check_overlap(sam_mask, black_mask):
            mask_result.append(torch.Tensor(sam_mask))
            box_result.append(torch.Tensor(tensor_sam_boxes[sam_mask_idx].detach().cpu().numpy()))

    if len(mask_result) == 0:
        return tensor_sam_masks, tensor_sam_boxes
    else:
        mask_result = torch.stack(mask_result)
        box_result = torch.stack(box_result)
        return mask_result, box_result

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            elif self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def check_overlap(mask1, mask2):
    return np.any(np.logical_and(mask1, mask2))

def merge_masks(masks):
    n = len(masks)
    uf = UnionFind(n)

    # 모든 마스크 쌍에 대해 겹치는지 확인하고, 겹친다면 연결
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap(masks[i], masks[j]):
                uf.union(i, j)
    
    # 같은 그룹의 마스크들을 합침
    merged_masks = {}
    for i in range(n):
        root = uf.find(i)
        if root in merged_masks:
            merged_masks[root] = np.logical_or(merged_masks[root], masks[i])
        else:
            merged_masks[root] = masks[i]
    
    return list(merged_masks.values())

def combine_masks(mask_list):
    # 입력된 리스트가 비어있지 않은지 확인
    if not mask_list:
        raise ValueError("The mask list is empty")

    # 모든 마스크를 numpy 배열로 변환
    mask_array = np.array(mask_list, dtype=int)

    # OR 연산을 수행하여 모든 마스크 합치기
    combined_mask = np.bitwise_or.reduce(mask_array)
    return combined_mask

def visualize_bounding_boxes(raw_img, image_size, boxes, k):
    """
    이미지에 바운딩 박스가 k번 겹치는 영역을 시각화합니다.

    :param image_size: 튜플, 이미지의 크기 (너비, 높이)
    :param boxes: 바운딩 박스 목록, 각 박스는 ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)) 형태
    :param k: 겹치는 횟수
    """
    # 이미지 크기에 맞는 빈 배열 생성
    overlay = np.zeros((image_size[1], image_size[0]))
    
    # 각 바운딩 박스에 대해 배열에 1을 더함으로써 겹치는 영역 계산
    for box in boxes:
        # PIL 이미지 좌표계에서 행렬 좌표계로 변환
        top_left_x, top_left_y = box[0]
        bottom_right_x, bottom_right_y = box[1]
        
        # 겹치는 영역에 해당하는 배열 부분에 1 더하기
        overlay[top_left_y:bottom_right_y, top_left_x:bottom_right_x] += 1

    # PIL 이미지 생성
    img = raw_img.copy()
    #if img.mode != 'RGBA':
    #    img = img.convert('RGBA')

    draw = ImageDraw.Draw(img)

    # 겹치는 횟수가 k인 영역을 찾아 시각화
    for y in range(image_size[1]):
        for x in range(image_size[0]):
            if overlay[y, x] < k:
                #draw.rectangle((x, y, x, y), outline=(0,0,0,0), fill=(0,0,255,50), width = 0) 
                draw.point((x, y), fill='black')

    return img

def generate_refer_masked_region_for_sam(model, preprocess, tokenizer, object_names:list, target_objects_list, raw_image: Image, device, overlap_count=3, kernel_size=200, stride = 50, draw_option=True):
    #모든 object names를 이용해서 나누고, 그 중에서 SAM에 사용할 것만 건져야한다.
    for target_objects in target_objects_list:
        for target_object in target_objects:
            if target_object not in object_names:
                raise KeyError(f"The target object({target_object}) is not included in object_names. The variable must include all of the possible object names.")
    #target_idxs_list = [[object_names.index(target_object) for target_object in target_objects] for target_objects in target_objects_list]
    

    target_idxs_list = [[object_names.index(target_object)] for target_object in object_names]
    
    points_list = [[] for _ in target_idxs_list]

    draw_image = raw_image.copy()
    width, height = raw_image.size

    num_x = len(range(0, width-kernel_size+1, stride))
    num_y = len(range(0,height-kernel_size+1, stride))
    object_texts = tokenizer(object_names)
    object_texts = object_texts.to(device)
    for x_idx, left_x in enumerate(range(0, width-kernel_size+1, stride)):
        for y_idx, left_y in enumerate(range(0,height-kernel_size+1, stride)):
            right_x = left_x + kernel_size
            right_y = left_y + kernel_size
            cropped_image = raw_image.crop((left_x, left_y, right_x, right_y))
            
            cropped_image = preprocess(cropped_image).unsqueeze(0)

            cropped_image = cropped_image.to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(cropped_image)
                text_features = model.encode_text(object_texts)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                object_align = torch.argmax(text_probs).detach().cpu().numpy()
                
                for idx, target_idxs in enumerate(target_idxs_list):
                    if object_align in target_idxs:
                        points_list[idx].append([(left_x, left_y), (right_x, right_y)])
                        if draw_option:
                            draw = ImageDraw.Draw(draw_image, "RGBA")
                            draw.rectangle((left_x, left_y, right_x, right_y), outline=(0,0,255,255), fill=(0,0,255,50), width = 3) 
    overlapped_images = []
    for points in points_list:
        overlapped_image = visualize_bounding_boxes(raw_image, raw_image.size, points, overlap_count)
        overlapped_images.append(overlapped_image)
        
    #overlapped_images = [visualize_bounding_boxes(raw_image, raw_image.size, points, overlap_count) for points in points_list]
    return draw_image, overlapped_images


def run_ris(clip_model, sam_model, raw_image: Image, masked_images, text_list, num_object_list, device="cuda"):
    image, w, h = preprocess(raw_image)
    mask_list = []
    torch.cuda.empty_cache()
    with torch.no_grad():

        #Generating SAM masks
        sam_masks = sam_model.generate(PIL2OpenCV(raw_image))
        sam_masks = sorted(sam_masks, key=lambda x: x["area"], reverse=True)
        tensor_sam_masks = torch.Tensor(np.stack([sam_masks[idx]["segmentation"] for idx in range(len(sam_masks))]))
        tensor_sam_masks = torch.Tensor.bool(tensor_sam_masks)
        tensor_sam_masks = tensor_sam_masks.to(device)
        tensor_sam_boxes = torch.Tensor([convert_sam_bbox_to_pil_bbox(sam_masks[idx]["bbox"]) for idx in range(len(sam_masks))])
        tensor_sam_boxes = tensor_sam_boxes.to(device)

        for mask_idx, masked_image in enumerate(masked_images):
            num_object = num_object_list[mask_idx]
            black_mask = extract_black_mask(masked_image)
            temp_sam_masks, temp_sam_boxes = filter_sam_mask(black_mask, tensor_sam_masks, tensor_sam_boxes)

            global_local_visual_features, crops = extract_global_local_visual_feature(input_img=image, clip_model=clip_model, 
                                        sam_masks=temp_sam_masks, sam_boxes=temp_sam_boxes, width=w, height=h, local_factor=0.75, crop_image_return=True)
            
            global_local_textual_feature = extract_global_local_text_feature(clip_model=clip_model, sentence=text_list[mask_idx])

            similarity = (100* global_local_visual_features @ global_local_textual_feature.T)

            sim_list = (similarity.clone().detach().cpu().numpy().flatten())
            sim_dict = {}
            for idx in range(len(sim_list)):
                sim_dict[idx] = sim_list[idx]

            sim_list = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)


            mask_1st_candidates = []
            mask_2nd_candidates = []
            mask_3rd_candidates = []
            mask_non_candidates = []

            for num_idx in range(len(temp_sam_masks)):
                mask_idx = sim_list[num_idx][0]
                if num_idx < num_object:
                    mask_1st_candidates.append(temp_sam_masks[mask_idx].detach().cpu().numpy())
                else:
                    mask_non_candidates.append(temp_sam_masks[mask_idx].detach().cpu().numpy())
            mask_1st_candidates = merge_masks(mask_1st_candidates)
            #mask_1st_candidates = merge_masks(mask_1st_candidates)
            print(f"First {len(mask_1st_candidates)} masks is decided")
                
            if len(mask_1st_candidates) < num_object: #complex_segmentation:
                #if len(mask_1st_candidates) < num_object:
                mask_2nd_candidates = mask_non_candidates[:num_object]
                mask_non_candidates = mask_non_candidates[num_object:]
                    
                mask_2nd_candidates = merge_masks(mask_1st_candidates+mask_2nd_candidates)

                print(f"Second {len(mask_2nd_candidates)} maks is decided")

                if len(mask_2nd_candidates) < num_object:
                    num_object = int(num_object*(2/3))
                    mask_3rd_candidates = mask_non_candidates[:num_object]
                    mask_non_candidates = mask_non_candidates[num_object:]
                    mask_3rd_candidates = merge_masks(mask_3rd_candidates+mask_2nd_candidates)
                    final_masks = mask_3rd_candidates
                    print(f"Third {len(mask_3rd_candidates)} masks is decided")
                
                elif len(mask_2nd_candidates) >= num_object:
                    mask_2nd_candidates = mask_2nd_candidates[:num_object]
                    final_masks = mask_2nd_candidates

            else:
                final_masks = mask_1st_candidates

            final_masks = [torch.Tensor.bool(torch.Tensor(mask)) for mask in final_masks]
            final_masks = combine_masks(final_masks)
            mask_list.append(final_masks)

    if len(mask_list) == 1:
        mask_list = mask_list[0]

    return mask_list


# def get_ris_models(device='cuda'):
#     #Load Third party CLIP for Referring Image Segmentation
#     clip_model, _ = clip.load('RN50')
#     clip_model= clip_model.to(device)
#     #Load HQ-SAM
#     sam_checkpoint = "pretrained_checkpoint/sam_hq_vit_l.pth"
#     sam_checkpoint = "/home/seungeon/Workspace/segment-anything/pretrained_checkpoint/sam_vit_h_4b8939.pth"
#     model_type = "vit_h"
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     mask_generator = SamAutomaticMaskGenerator(
#         model=sam,
#         points_per_side=24,
#         pred_iou_thresh=0.8,
#         stability_score_thresh=0.9,
#         crop_n_layers=1,
#         crop_n_points_downscale_factor=2,
#         #min_mask_region_area=224*224,  # Requires open-cv to run post-processing
#     )
#     return clip_model, mask_generator

def arange_masked_images(masked_images, object_names, target_objects_list):
    target_idxs_list = [[object_names.index(target_object) for target_object in target_objects] for target_objects in target_objects_list]
    num_case = len(target_objects_list)
    result = [[] for _ in range(num_case)]

    for result_idx, target_idxs in enumerate(target_idxs_list):

        for target_idx in target_idxs:
            result[result_idx].append(masked_images[target_idx])
    return result

