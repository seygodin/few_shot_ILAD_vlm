import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from . import my_clip

from .loralib import layers as lora_layers
from .loralib import mark_only_lora_as_trainable

from PIL import Image, ImageDraw
import numpy as np
import copy
from time import time
class feature_encoder(nn.Module):
    def __init__(self, in_dim, out_dim, lora):
        super(feature_encoder, self).__init__()

        self.encoder = nn.Sequential(lora_layers.Linear(in_features=in_dim, out_features=in_dim//5, r=lora),
                                    #nn.BatchNorm1d(in_dim//5),
                                    nn.ReLU(),
                                    lora_layers.Linear(in_features=in_dim//5, out_features=in_dim//10, r=lora),
                                    #nn.BatchNorm1d(5*768),
                                    nn.ReLU(),
                                    lora_layers.Linear(in_features=in_dim//10, out_features=out_dim, r=lora),)

    def forward(self, input_feature):
        return self.encoder(input_feature)
    

class cnn_feature_encoder(nn.Module):
    def __init__(self, lora, num_channel=768, in_dim = 256*7*7, out_dim=512):
        super(cnn_feature_encoder, self).__init__()
        self.num_channel = num_channel
        self.cnn_encoder = nn.Sequential(lora_layers.Conv2d(in_channels=num_channel, out_channels=num_channel//2, kernel_size=3, stride=1, padding=1),
                                         nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         lora_layers.Conv2d(in_channels=num_channel//2, out_channels=num_channel//3, kernel_size=3, stride=1, padding=1),
                                         nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         lora_layers.Conv2d(in_channels=num_channel//3, out_channels=num_channel//3, kernel_size=3, stride=1, padding=1),
                                                )
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp_head = nn.Sequential(lora_layers.Linear(in_features=self.in_dim, out_features=self.in_dim//2, r=lora),
                                      nn.ReLU(),
                                      lora_layers.Linear(in_features=self.in_dim//2, out_features=self.in_dim//4, r=lora),
                                      nn.ReLU(),
                                      lora_layers.Linear(in_features=self.in_dim//4, out_features=self.out_dim, r=lora),
                                      )
        
    def forward(self, input_feature):
        #Remove class token
        input_feature = input_feature[:,:49,:]
        b, _, c = input_feature.shape
        input_feature = input_feature.reshape(b, 7, 7, c)
        input_feature = input_feature.permute(0, 3, 1, 2)

        cnn_output = self.cnn_encoder(input_feature)

        b, c, h, w = cnn_output.shape

        cnn_output = cnn_output.reshape(b, c*h*w)

        mlp_output = self.mlp_head(cnn_output)

        return mlp_output



class ad_clip(nn.Module):
    #def __init__(self, pretrained_clip, clip_preprocess, clip_loss, lora = 1, input_dim=50*768, hidden_dim=512):
    def __init__(self, pretrained_clip, clip_preprocess, clip_loss, lora = 1, input_dim=88*257, hidden_dim=1024):
        super(ad_clip, self).__init__()
        self.clip: nn.Module = pretrained_clip
        self.clip_preprocess = clip_preprocess
        self.lora = lora
        self.lora_set = False

        self.hidden_dim = hidden_dim
        self.clip_loss = clip_loss

        self.flat = nn.Flatten(start_dim=1, end_dim=2)
        self.only_decoder = False
        self.pooling= nn.AvgPool1d(kernel_size=16, stride=16)

        self.feature_layer1 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)
        self.feature_layer2 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)
        self.feature_layer3 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)
        self.feature_layer4 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)
        self.feature_layer5 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)

        self.decoder_layer1 = nn.Sequential(lora_layers.Linear(in_features=hidden_dim, out_features=hidden_dim, r=lora),
                                            #nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(),
                                            lora_layers.Linear(in_features=hidden_dim, out_features=hidden_dim, r=lora))
        
        self.decoder_layer2 = nn.Sequential(lora_layers.Linear(in_features=hidden_dim*6, out_features=hidden_dim*3, r=lora),
                                            #nn.BatchNorm1d(hidden_dim*3),
                                            nn.ReLU(),
                                            lora_layers.Linear(in_features=hidden_dim*3, out_features=hidden_dim*1, r=lora),
                                            #nn.BatchNorm1d(hidden_dim*1),
                                            nn.ReLU(),
                                            lora_layers.Linear(in_features=hidden_dim*1, out_features=hidden_dim, r=lora),)
        
    def forward(self, image, text, return_loss=False):

        
        image_features, mid_features = self.base_encode_image(image,return_feature=True)
        text_features = self.encode_text(text)
        feature1 = self.feature_layer1(self.flat(mid_features[-1]))
        feature2 = self.feature_layer2(self.flat(mid_features[-2]))
        feature3 = self.feature_layer3(self.flat(mid_features[-3]))
        feature4 = self.feature_layer4(self.flat(mid_features[-4]))
        feature5 = self.feature_layer5(self.flat(mid_features[-5]))

        image_features = self.decoder_layer1(image_features)

        total_feature = torch.cat((image_features, feature1, feature2, feature3, feature4, feature5), dim=1)

        final_image_feature = self.decoder_layer2(total_feature)


        if return_loss:
            losses = self.clip_loss(final_image_feature, text_features, self.clip.logit_scale.exp())
            return final_image_feature, text_features, losses

        else:
            return final_image_feature, text_features
        



    def base_encode_image(self, image, return_feature=False):
        #image = self.clip_preprocess(image)
        # image = self.image_encoder_layer1(image)
        # image = self.image_encoder_layer2(image)
        image = self.clip.encode_image(image, normalize=True) # [B,3,W,H]

        
            
        if return_feature:
            step_features = []
            for i in range(len(self.clip.visual.transformer.resblocks)):
                step_feature = self.clip.visual.transformer.resblocks[i].step_image_feature.permute(1,0,2)
                if self.hidden_dim == 1024:
                    step_feature = self.pooling(step_feature)
                #print(step_feature.shape)
                step_features.append(step_feature)

            return image, step_features
            
        return image
    
    def encode_image(self, image):
        image_features = self.clip.encode_image(image, normalize=False)

        step_features = []
        for i in range(len(self.clip.visual.transformer.resblocks)):
            step_feature = self.clip.visual.transformer.resblocks[i].step_image_feature.permute(1,0,2)

            if self.hidden_dim == 1024:
                step_feature = self.pooling(step_feature)
            step_features.append(step_feature)

        feature1 = self.feature_layer1(self.flat(step_features[-1]))
        feature2 = self.feature_layer2(self.flat(step_features[-2]))
        feature3 = self.feature_layer3(self.flat(step_features[-3]))
        feature4 = self.feature_layer4(self.flat(step_features[-4]))
        feature5 = self.feature_layer5(self.flat(step_features[-5]))

        image_features = self.decoder_layer1(image_features)

        total_feature = torch.cat((image_features, feature1, feature2, feature3, feature4, feature5), dim=1)

        final_image_feature = self.decoder_layer2(total_feature)

        return final_image_feature

    
    def encode_text(self, text):
        text_features = self.clip.encode_text(text, normalize=True)
        return text_features
    


class ad_clip2(nn.Module):
    #def __init__(self, pretrained_clip, clip_preprocess, clip_loss, lora = 1, input_dim=50*768, hidden_dim=512):
    def __init__(self, pretrained_clip, clip_preprocess, clip_loss, lora = 1, input_dim=88*257, hidden_dim=1024):
        super(ad_clip2, self).__init__()
        self.clip: nn.Module = pretrained_clip
        self.mask_image_clip: nn.Module = copy.deepcopy(pretrained_clip)

        self.clip_preprocess = clip_preprocess

        self.lora = lora
        self.lora_set = False

        self.hidden_dim = hidden_dim
        self.clip_loss = clip_loss

        self.flat = nn.Flatten(start_dim=1, end_dim=2)
        self.only_decoder = False
        self.pooling= nn.AvgPool1d(kernel_size=16, stride=16)

        self.feature_layer1 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)
        self.feature_layer2 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)
        self.feature_layer3 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)
        self.feature_layer4 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)
        self.feature_layer5 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)

        self.decoder_layer1 = nn.Sequential(lora_layers.Linear(in_features=hidden_dim, out_features=hidden_dim, r=lora),
                                            #nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(),
                                            lora_layers.Linear(in_features=hidden_dim, out_features=hidden_dim, r=lora))
        
        self.decoder_layer2 = nn.Sequential(lora_layers.Linear(in_features=hidden_dim*6, out_features=hidden_dim*3, r=lora),
                                            #nn.BatchNorm1d(hidden_dim*3),
                                            nn.ReLU(),
                                            lora_layers.Linear(in_features=hidden_dim*3, out_features=hidden_dim*1, r=lora),
                                            #nn.BatchNorm1d(hidden_dim*1),
                                            nn.ReLU(),
                                            lora_layers.Linear(in_features=hidden_dim*1, out_features=hidden_dim, r=lora),)
        

        self.mask_feature_layer1 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)
        self.mask_feature_layer2 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)
        self.mask_feature_layer3 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)
        self.mask_feature_layer4 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)
        self.mask_feature_layer5 = feature_encoder(in_dim=input_dim, out_dim=hidden_dim, lora=lora)
        
        self.mask_decoder_layer1 = nn.Sequential(lora_layers.Linear(in_features=hidden_dim, out_features=hidden_dim, r=lora),
                                            #nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(),
                                            lora_layers.Linear(in_features=hidden_dim, out_features=hidden_dim, r=lora))
        
        self.mask_decoder_layer2 = nn.Sequential(lora_layers.Linear(in_features=hidden_dim*6, out_features=hidden_dim*3, r=lora),
                                            #nn.BatchNorm1d(hidden_dim*3),
                                            nn.ReLU(),
                                            lora_layers.Linear(in_features=hidden_dim*3, out_features=hidden_dim*1, r=lora),
                                            #nn.BatchNorm1d(hidden_dim*1),
                                            nn.ReLU(),
                                            lora_layers.Linear(in_features=hidden_dim*1, out_features=hidden_dim, r=lora),)

    def forward(self, image, text, return_loss=False):

        
        image_features, mid_features = self.base_encode_image(image,return_feature=True)
        text_features = self.encode_text(text)
        feature1 = self.feature_layer1(self.flat(mid_features[-1]))
        feature2 = self.feature_layer2(self.flat(mid_features[-2]))
        feature3 = self.feature_layer3(self.flat(mid_features[-3]))
        feature4 = self.feature_layer4(self.flat(mid_features[-4]))
        feature5 = self.feature_layer5(self.flat(mid_features[-5]))

        image_features = self.decoder_layer1(image_features)

        total_feature = torch.cat((image_features, feature1, feature2, feature3, feature4, feature5), dim=1)

        final_image_feature = self.decoder_layer2(total_feature)


        if return_loss:
            losses = self.clip_loss(final_image_feature, text_features, self.clip.logit_scale.exp())
            return final_image_feature, text_features, losses

        else:
            return final_image_feature, text_features
        



    def base_encode_image(self, image, return_feature=False):
        #image = self.clip_preprocess(image)
        # image = self.image_encoder_layer1(image)
        # image = self.image_encoder_layer2(image)
        image = self.clip.encode_image(image, normalize=True) # [B,3,W,H]

        
            
        if return_feature:
            step_features = []
            for i in range(len(self.clip.visual.transformer.resblocks)):
                step_feature = self.clip.visual.transformer.resblocks[i].step_image_feature.permute(1,0,2)
                if self.hidden_dim == 1024:
                    step_feature = self.pooling(step_feature)
                #print(step_feature.shape)
                step_features.append(step_feature)

            return image, step_features
            
        return image
    
    def encode_image(self, image):
        image_features = self.clip.encode_image(image, normalize=False)

        step_features = []
        for i in range(len(self.clip.visual.transformer.resblocks)):
            step_feature = self.clip.visual.transformer.resblocks[i].step_image_feature.permute(1,0,2)

            if self.hidden_dim == 1024:
                step_feature = self.pooling(step_feature)
            step_features.append(step_feature)

        feature1 = self.feature_layer1(self.flat(step_features[-1]))
        feature2 = self.feature_layer2(self.flat(step_features[-2]))
        feature3 = self.feature_layer3(self.flat(step_features[-3]))
        feature4 = self.feature_layer4(self.flat(step_features[-4]))
        feature5 = self.feature_layer5(self.flat(step_features[-5]))

        image_features = self.decoder_layer1(image_features)

        total_feature = torch.cat((image_features, feature1, feature2, feature3, feature4, feature5), dim=1)

        final_image_feature = self.decoder_layer2(total_feature)

        return final_image_feature
    
    def encode_mask_image(self, image):
        image_features = self.mask_image_clip.encode_image(image, normalize=False)

        step_features = []
        for i in range(len(self.mask_image_clip.visual.transformer.resblocks)):
            step_feature = self.mask_image_clip.visual.transformer.resblocks[i].step_image_feature.permute(1,0,2)

            if self.hidden_dim == 1024:
                step_feature = self.pooling(step_feature)
            step_features.append(step_feature)

        feature1 = self.mask_feature_layer1(self.flat(step_features[-1]))
        feature2 = self.mask_feature_layer2(self.flat(step_features[-2]))
        feature3 = self.mask_feature_layer3(self.flat(step_features[-3]))
        feature4 = self.mask_feature_layer4(self.flat(step_features[-4]))
        feature5 = self.mask_feature_layer5(self.flat(step_features[-5]))

        image_features = self.mask_decoder_layer1(image_features)

        total_feature = torch.cat((image_features, feature1, feature2, feature3, feature4, feature5), dim=1)

        final_image_feature = self.mask_decoder_layer2(total_feature)

        return final_image_feature


        #return self.mask_image_clip.encode_image(image)

    
    def encode_text(self, text):
        text_features = self.clip.encode_text(text, normalize=True)
        return text_features
    




class normal_clip(nn.Module):
    def __init__(self, pretrained_clip, clip_preprocess, clip_loss):
        super(normal_clip, self).__init__()
        self.clip: nn.Module = pretrained_clip
        self.clip_preprocess = clip_preprocess
        self.clip_loss = clip_loss

    def forward(self, image, text, return_loss=False):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        if return_loss:
            losses = self.clip_loss(image_features, text_features, self.clip.logit_scale.exp())
            return image_features, text_features, losses
        else:
            return image_features, text_features

    def base_encode_image(self, image, return_feature=False):
        image = self.clip.encode_image(image, normalize=True) # [B,3,W,H]
        return image
    
    def encode_image(self, image):
        image = self.clip.encode_image(image, normalize=False)
        return image

    def encode_text(self, text):
        text_features = self.clip.encode_text(text, normalize=True)
        return text_features
    

def get_model(args, model_name='ViT-B-32', pretrained_name='laion2b_s34b_b79k', default=False, double_encoder=False):
    

    if args.foundation == "CLIP":
        model, _, preprocess = my_clip.create_model_and_transforms(args=args, model_name=model_name, pretrained=pretrained_name)

        #model, _, preprocess = my_clip.create_model_and_transforms('EVA02-E-14-plus', pretrained='laion2b_s9b_b144k')
    
        tokenizer = my_clip.get_tokenizer(model_name)

        clip_loss = my_clip.loss.ClipLoss()

        if default:
            my_model = normal_clip(pretrained_clip=model, clip_preprocess=preprocess, clip_loss=clip_loss)

        elif model_name == "ViT-B-32":
            if args.pretrained_count_path != None:
                model.load_state_dict(torch.load(args.pretrained_count_path), strict=False)

            if double_encoder != 1:
                my_model = ad_clip(pretrained_clip=model, clip_preprocess=preprocess, clip_loss=clip_loss, lora=1, input_dim=50*768, hidden_dim=512)
            else:
                my_model = ad_clip2(pretrained_clip=model, clip_preprocess=preprocess, clip_loss=clip_loss, lora=1, input_dim=50*768, hidden_dim=512)
            #my_model = ad_clip2(pretrained_clip=model, clip_preprocess=preprocess, clip_loss=clip_loss, lora=1, input_dim=50*768, hidden_dim=512)
        elif model_name == 'ViT-L-14':
            raise NotImplementedError("The model for other pretrained model is not implemented.")
            #my_model = ad_clip(pretrained_clip=model, clip_preprocess=preprocess, clip_loss=clip_loss, lora=1, input_dim=263168)
        elif model_name == 'ViT-g-14':
            if double_encoder != 1:
                my_model = ad_clip(pretrained_clip=model, clip_preprocess=preprocess, clip_loss=clip_loss, lora=1, input_dim=88*257, hidden_dim=1024)
            else:
                my_model = ad_clip2(pretrained_clip=model, clip_preprocess=preprocess, clip_loss=clip_loss, lora=1, input_dim=88*257, hidden_dim=1024)
        else:
            raise NotImplementedError("The model for other pretrained model is not implemented.")
        
        #optimizer = optim.Adamw(params=my_model.parameters(), lr=args.lr)
        optimizer = optim.AdamW(params=my_model.parameters(), lr=args.lr, betas=(0.9, 0.999))

        if not default:
            mark_only_lora_as_trainable(model = my_model)
        else:
            pass

        return my_model, preprocess, tokenizer, optimizer
    
    elif args.foundation == "BLIP":
        raise NotImplementedError
    elif args.foundation == "BLIP2":
        raise NotImplementedError
    
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

def generate_refer_masked_image_batch(
    model,
    preprocess,
    tokenizer,
    object_names: list,
    target_objects_list,
    raw_image: Image,
    device,
    overlap_count=3,
    kernel_size=200,
    stride=50,
    draw_option=True,
    batch_size=256,
):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    for target_objects in target_objects_list:
        for target_object in target_objects:
            if target_object not in object_names:
                raise KeyError(
                    f"The target object({target_object}) is not included in object_names. The variable must include all of the possible object names."
                )
    target_idxs_list = [
        [object_names.index(target_object) for target_object in target_objects]
        for target_objects in target_objects_list
    ]
    points_list = [[] for _ in target_idxs_list]

    draw_image = raw_image.copy()
    width, height = raw_image.size

    num_x = len(range(0, width - kernel_size + 1, stride))
    num_y = len(range(0, height - kernel_size + 1, stride))
    object_texts = tokenizer(object_names)
    object_texts = object_texts.to(device)

    slide_points = [
        (left_x, left_y, left_x + kernel_size, left_y + kernel_size)
        for left_x in range(0, width - kernel_size + 1, stride)
        for left_y in range(0, height - kernel_size + 1, stride)
    ]

    with torch.no_grad():
        text_features = model.encode_text(object_texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    for start_idx in range(0, len(slide_points), batch_size):
        batch_points = slide_points[start_idx : start_idx + batch_size]

        cropped_images = []
        for left_x, left_y, right_x, right_y in batch_points:
            cropped_image = raw_image.crop((left_x, left_y, right_x, right_y))
            cropped_images.append(preprocess(cropped_image))

        cropped_images = torch.stack(cropped_images, dim=0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(cropped_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        object_aligns = torch.argmax(text_probs, dim=-1).detach().cpu().tolist()

        for point, object_align in zip(batch_points, object_aligns):
            left_x, left_y, right_x, right_y = point
            for idx, target_idxs in enumerate(target_idxs_list):
                if object_align in target_idxs:
                    points_list[idx].append([(left_x, left_y), (right_x, right_y)])
                    if draw_option:
                        draw = ImageDraw.Draw(draw_image, "RGBA")
                        draw.rectangle(
                            (left_x, left_y, right_x, right_y),
                            outline=(0, 0, 255, 255),
                            fill=(0, 0, 255, 50),
                            width=3,
                        )

    # Keep existing variables for logic compatibility/debug parity.
    _ = num_x, num_y

    overlapped_images = []
    for points in points_list:
        overlapped_image = visualize_bounding_boxes(
            raw_image, raw_image.size, points, overlap_count
        )
        overlapped_images.append(overlapped_image)

    return draw_image, overlapped_images


def generate_refer_masked_image(
    model,
    preprocess,
    tokenizer,
    object_names: list,
    target_objects_list,
    raw_image: Image,
    device,
    overlap_count=3,
    kernel_size=200,
    stride=50,
    draw_option=True,
):
    for target_objects in target_objects_list:
        for target_object in target_objects:
            if target_object not in object_names:
                raise KeyError(
                    f"The target object({target_object}) is not included in object_names. The variable must include all of the possible object names."
                )
    target_idxs_list = [
        [object_names.index(target_object) for target_object in target_objects]
        for target_objects in target_objects_list
    ]
    points_list = [[] for _ in target_idxs_list]

    draw_image = raw_image.copy()
    width, height = raw_image.size

    num_x = len(range(0, width - kernel_size + 1, stride))
    num_y = len(range(0, height - kernel_size + 1, stride))
    object_texts = tokenizer(object_names)
    object_texts = object_texts.to(device)

    for x_idx, left_x in enumerate(range(0, width - kernel_size + 1, stride)):
        for y_idx, left_y in enumerate(range(0, height - kernel_size + 1, stride)):
            right_x = left_x + kernel_size
            right_y = left_y + kernel_size
            cropped_image = raw_image.crop((left_x, left_y, right_x, right_y))

            cropped_image = preprocess(cropped_image).unsqueeze(0)
            cropped_image = cropped_image.to(device)

            with torch.no_grad():
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
                            draw.rectangle(
                                (left_x, left_y, right_x, right_y),
                                outline=(0, 0, 255, 255),
                                fill=(0, 0, 255, 50),
                                width=3,
                            )

    _ = num_x, num_y, x_idx, y_idx

    overlapped_images = []
    for points in points_list:
        overlapped_image = visualize_bounding_boxes(
            raw_image, raw_image.size, points, overlap_count
        )
        overlapped_images.append(overlapped_image)

    return draw_image, overlapped_images
