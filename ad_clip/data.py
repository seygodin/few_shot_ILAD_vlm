import torch.utils.data as data
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from .model import generate_refer_masked_image, generate_refer_masked_image_batch
#from .ris import get_ris_models
from .ris import generate_refer_masked_region_for_sam, run_ris, arange_masked_images, combine_masks
from .utils_save import extract_feature_from_masked_image, extract_masks_from_seg_image, arange_seg_idx_to_target_list
import open_clip

y_format = {"normal sa la": "1 0 0 or 0 1 0 or 0 0 1"}

def find_unique_strings(double_list):
    unique_strings = set()  # 고유한 문자열을 저장할 세트 생성
    for sublist in double_list:
        for item in sublist:
            unique_strings.add(item)  # 각 문자열을 세트에 추가하여 고유하게 유지
    return list(unique_strings)  # 세트를 리스트로 변환하여 반환

def zero_max_pad(img, is_mask=False):
    h, w = img.size
    if h >= w:
        size = h
    else:
        size = w

    if is_mask:
        new_arr = np.zeros((size, size),  dtype=np.uint8)
        arr = np.array(img)
        w, h = arr.shape
        new_arr[:w, :h] = arr
    else:
        new_arr = np.zeros((size, size, 3), dtype=np.uint8)
        arr = np.array(img)
        w, h, _ = arr.shape
        new_arr[:w, :h] = arr
    return Image.fromarray(new_arr)

def apply_bi_mask_to_image(origin_image, binary_mask, display_option=False):
    if isinstance(origin_image, np.ndarray):
        origin_image = Image.fromarray(origin_image)

    binary_mask = binary_mask.astype(np.uint8)*255
    if isinstance(binary_mask, np.ndarray):
        #binary_mask = np.transpose(binary_mask)
        binary_mask_image = Image.fromarray(binary_mask)

    
    if display_option:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(origin_image)

        plt.subplot(1,2,2)
        plt.imshow(binary_mask_image)

    masked_image = Image.composite(origin_image, Image.new('RGB', origin_image.size), binary_mask_image)
    
    return masked_image


class ad_dataset(data.Dataset):
    def __init__(self, data_path_dict: dict, path_name: str, preprocess, level:int, device="cpu"):
        self.image_list = [preprocess(zero_max_pad(Image.open(img_path))) for img_path in data_path_dict[path_name]]
        self.img_paths = [img_path for img_path in data_path_dict[path_name]]
        self.image_tensor = torch.stack(self.image_list, dim=0).to(device)
        self.length = len(self.image_list)

        self.level = level

        #self.candidates = [i for i in range(0, len(self.pos_rule))]

    def __getitem__(self, index):
        tensor_img = self.image_tensor[index]
        return tensor_img

    def __len__(self):
        return self.length
    

class ad_obj_dataset(data.Dataset):
    def __init__(self, data_path_dict: dict, path_name: str, model, preprocess, tokenizer, 
                 device, object_names, target_objects_list, overlap_count, kernel_size, stride, save=False, save_path=None):
        #print(data_path_dict[path_name])
        #self.image_list = [preprocess(Image.open(img_path).resize((224,224))) for img_path in data_path_dict[path_name]]
        #self.image_tensor = torch.stack(self.image_list, dim=0).to(device)
        #self.length = len(self.image_list)
        


        self.image_list = []
        self.img_paths = []
        self.masked_img = []
        print(device)
        model = model.to(device)
        for img_path in data_path_dict[path_name]:
            temp_img = Image.open(img_path)
            padded_temp_img = zero_max_pad(temp_img)
            padded_temp_img = temp_img
            print(img_path, padded_temp_img.size)
            self.image_list.append(preprocess(padded_temp_img))
            # _, masked_imgs = generate_refer_masked_image(model, preprocess, tokenizer, object_names, target_objects_list, 
            #                                              temp_img, device, overlap_count=overlap_count, kernel_size=kernel_size, stride = stride, 
            #                                              draw_option=True)
            _, masked_imgs = generate_refer_masked_image_batch(model, preprocess, tokenizer, object_names, target_objects_list, 
                                                         temp_img, device, overlap_count=overlap_count, kernel_size=kernel_size, stride = stride, 
                                                         draw_option=True)
            masked_imgs = [zero_max_pad(masked_img) for masked_img in masked_imgs]
            #for maksed_img in masked_imgs:
            #    print(maksed_img.size, type(maksed_img))
            if save:
                img_name = os.path.basename(img_path).split(".")[0]
                save_img_dir = os.path.join(save_path, img_name)

                if not os.path.exists(save_img_dir):
                    #os.mkdir(save_img_dir)
                    os.makedirs(save_img_dir, exist_ok=True)

                padded_temp_img.save(os.path.join(save_img_dir, "origin.png"))

                for mask_idx, masked_img in enumerate(masked_imgs):
                    mask_img_name = f"image_with_{target_objects_list[mask_idx]}_mask_{mask_idx}.png"
                    masked_img.save(os.path.join(save_img_dir, mask_img_name))

            masked_imgs = [preprocess(masked_img) for masked_img in masked_imgs]
            masked_imgs = torch.stack(masked_imgs, dim=0)
            self.masked_img.append(masked_imgs)
            self.img_paths.append(img_path)

        self.image_tensor = torch.stack(self.image_list, dim=0).to(device)
        self.masked_img = torch.stack(self.masked_img, dim=0).to(device)
        self.length = len(self.image_list)

        print(f"Maksed data: {self.masked_img.shape}")
        print(f"Raw Image data: {self.image_tensor.shape}")

    def __getitem__(self, index):
        tensor_img = self.image_tensor[index]
        masked_img = self.masked_img[index]
        return tensor_img, masked_img

    def __len__(self):
        return self.length


#Using augmented data. 
class ad_aug_dataset(data.Dataset):
    def __init__(self, data_name: str, data_path_dict: dict, path_name: str, model, preprocess, tokenizer, 
                 device, object_names, target_objects_list, save=False, save_path=None):
        #print(data_path_dict[path_name])
        #self.image_list = [preprocess(Image.open(img_path).resize((224,224))) for img_path in data_path_dict[path_name]]
        #self.image_tensor = torch.stack(self.image_list, dim=0).to(device)
        #self.length = len(self.image_list)
        


        self.image_list = []
        self.img_paths = []
        self.masked_img = []
        
        self.meta1 = []
        self.meta2 = []
        print(device)
        model = model.to(device)

        index_list, supposed_num = arange_seg_idx_to_target_list(data_name)

        
        for img_path in data_path_dict[path_name]:
            real_img = Image.open(img_path)
            real_img = zero_max_pad(real_img)
            real_img = real_img.resize(size=(512,512))

            self.image_list.append(preprocess(real_img))

            seg_img_path = img_path.replace("orig", "orig_seg")
            seg_img = Image.open(seg_img_path)  #Size of seg_img is (512, 512)
            np_seg_img = np.array(seg_img)
            seg_mask_list = extract_masks_from_seg_image(seg_img, supposed_num)

            print(img_path, real_img.size, seg_img.size)

            target_mask_list = []

            for target_list in target_objects_list:
                temp_mask_list = []
                for target in target_list:
                    target_idx = object_names.index(target)
                    print(f"({target}) -> {index_list[target_idx]}")
                    mask_numbers = index_list[target_idx]

                    for mask_number in mask_numbers:
                        temp_mask_list.append(seg_mask_list[mask_number])
                        
                    temp_mask = combine_masks(temp_mask_list)
                    temp_mask_list.append(temp_mask)
                final_mask = combine_masks(temp_mask_list)
                target_mask_list.append(final_mask)

            masked_image_list = []
            mask_num_pixel_list = []
            mask_act_feature_list = []
            
            for target_mask_idx, mask in enumerate(target_mask_list):
                np_real_img = np.array(real_img)
                np_real_img[np.logical_not(mask)] = 0
                
                num_zero = len(np_real_img[np_real_img==0])
                num_non_zero = len(np_real_img[np_real_img!=0])

                if num_zero == 0:
                    num_zero = 1
                if num_non_zero ==0:
                    num_non_zero = 1
                num_act_pixel = len(np_real_img[np_real_img!=0])
                seg_feature = np.sum(np_real_img)/(np_real_img.size - len(np_real_img[np_real_img==0]))
                mask_num_pixel_list.append(num_act_pixel)
                mask_act_feature_list.append(seg_feature)
                
                masked_real_image = Image.fromarray(np_real_img)
                masked_image_list.append(masked_real_image)

            if save:
                img_name = os.path.basename(img_path).split(".")[0]
                save_img_dir = os.path.join(save_path, img_name)

                if not os.path.exists(save_img_dir):
                    #os.mkdir(save_img_dir)
                    os.makedirs(save_img_dir)

                real_img.save(os.path.join(save_img_dir, "origin.png"))

                for mask_idx, masked_img in enumerate(w):
                    mask_img_name = f"image_with_{target_objects_list[mask_idx]}_mask_{mask_idx}.png"
                    masked_img.save(os.path.join(save_img_dir, mask_img_name))

            masked_imgs = [preprocess(masked_img) for masked_img in masked_image_list]
            masked_imgs = torch.stack(masked_imgs, dim=0)
            
            mask_num_pixel_list = torch.Tensor(mask_num_pixel_list)
            mask_act_feature_list = torch.Tensor(mask_act_feature_list)
            
            self.masked_img.append(masked_imgs)
            self.img_paths.append(img_path)
            self.meta1.append(mask_num_pixel_list)
            self.meta2.append(mask_act_feature_list)

        self.image_tensor = torch.stack(self.image_list, dim=0).to(device)
        self.masked_img = torch.stack(self.masked_img, dim=0).to(device)
        #self.meta1 = torch.stack(self.meta1, dim=0).to(device)
        #self.meta2 = torch.stack(self.meta2, dim=0).to(device)
        self.meta1 = F.normalize(torch.stack(self.meta1, dim=0)).to(device)
        self.meta2 = F.normalize(torch.stack(self.meta2, dim=0)).to(device)
        
        self.length = len(self.image_list)

        print(f"Maksed data: {self.masked_img.shape}")
        print(f"Raw Image data: {self.image_tensor.shape}")
        print(f"Mask Meta info: {self.meta1.shape}, {self.meta2.shape}")

    def __getitem__(self, index):
        tensor_img = self.image_tensor[index]
        masked_img = self.masked_img[index]
        #return tensor_img, masked_img
        
        #revised
        meta1 = self.meta1[index]
        meta2 = self.meta2[index]
        return tensor_img, masked_img, meta1.float(), meta2.float()

    def __len__(self):
        return self.length
#Dataset for using augmented data
    
def sam_info_check(object_names, sam_text_list, target_object_list):
    if len(sam_text_list) > len(object_names):
        raise ValueError("The number of sam_text_list cannot be bigger than the number of object_names")
    sam_target_objects = object_names[:len(sam_text_list)]

    unique_target_list = []

    for i in target_object_list:
        for k in i:
            if k in unique_target_list:
                pass
            elif k not in unique_target_list:
                unique_target_list.append(k)

    for target in unique_target_list:
        if target not in sam_target_objects:
            raise KeyError(f"Every object in target object list must be the target of RIS-SAM. '{target}' is missing")
    
    return True
"""
class ad_sam_ris_dataset(data.Dataset):
    def __init__(self, data_path_dict: dict, path_name: str, model, preprocess, tokenizer, 
                 device, object_names, target_objects_list, sam_text_list, sam_num_list, overlap_count, kernel_size, stride, save=False, save_path=None):
        #sam_text_list와 sam_num_list는 object_names와 같은 순서로 주어지는 SAM에 사용하기 위한 정보들
        #Example: Juice bottle
        #object_names = ["a banana picture label", "a 100% text label",  "a bottle cap", "a black background"]
        #sam_text_list= ["a fruit picture label", "a 100% text label",  "a bottle cap"],
        #sam_num_list = [1, 1, 1]
        #target_objects_list = [
        #    ["a banana picture label"],
        #    ["a banana picture label", "a 100% text label"],
        #    ["a banana picture label"],
        #    ["a 100% text label"],
        #    ["a banana picture label", "a 100% text label"]
        #]
 
        self.image_list = []
        self.img_paths = []
        self.masked_img = []
        model = model.to(device)

        self.sam_mask_list = []

        self.object_names = object_names
        self.meta_feature = []

        sam_info_check(object_names, sam_text_list, target_objects_list)

        for img_path in data_path_dict[path_name]:
            object_names = self.object_names
            rule_mask_list = []
            temp_meta_feature = []
            
            temp_img = Image.open(img_path)
            padded_temp_img = zero_max_pad(temp_img)
            print(img_path, padded_temp_img.size)
            
            self.image_list.append(preprocess(padded_temp_img))

            img_name = os.path.basename(img_path).split(".")[0]
            save_img_dir = os.path.join(save_path, img_name)
            if not os.path.exists(save_img_dir):
                os.mkdir(save_img_dir)
            padded_temp_img.save(os.path.join(save_img_dir, "origin.png"))
            sam_mask_list = []


            with torch.no_grad():
                torch.cuda.empty_cache()
                clip_model, sam_model = get_ris_models(device)
                sam_mask = run_ris(clip_model, sam_model, raw_image=padded_temp_img, masked_images=masked_imgs, 
                                    text_list=sam_text_list, num_object_list=sam_num_list, device="cuda")
                sam_mask_list = sam_mask


            rule_sam_mask_list = [ [] for _ in range(len(target_objects_list))]
            rule_sam_masked_image_list = []
            for rule_idx, target_list in enumerate(target_objects_list):
                for target in target_list:
                    mask_idx = object_names.index(target)
                    selected_mask = sam_mask_list[mask_idx]
                    rule_sam_mask_list[rule_idx].append(selected_mask)
                rule_sam_mask_list[rule_idx] = combine_masks(rule_sam_mask_list[rule_idx])
                masked_img = apply_bi_mask_to_image(padded_temp_img, rule_sam_mask_list[rule_idx], display_option=False)
                rule_sam_masked_image_list.append(masked_img)
                meta_feature = extract_feature_from_masked_image(masked_img)
                temp_meta_feature.append(meta_feature)
            
            self.sam_mask_list.append(rule_sam_mask_list)

            if save:
                img_name = os.path.basename(img_path).split(".")[0]
                save_img_dir = os.path.join(save_path, img_name)

                if not os.path.exists(save_img_dir):
                    os.mkdir(save_img_dir)

                padded_temp_img.save(os.path.join(save_img_dir, "origin.png"))

                for mask_idx, masked_img in enumerate(rule_sam_masked_image_list):
                    mask_img_name = f"ris_image_with_{target_objects_list[mask_idx]}_mask_{mask_idx}.png"
                    masked_img.save(os.path.join(save_img_dir, mask_img_name))

            masked_imgs = [preprocess(masked_img) for masked_img in rule_sam_masked_image_list]
            masked_imgs = torch.stack(masked_imgs, dim=0)
            
            temp_meta_feature = torch.stack(temp_meta_feature, dim=0)
            
            self.masked_img.append(masked_imgs)
            self.img_paths.append(img_path)
            
            self.meta_feature.append(temp_meta_feature)

        self.image_tensor = torch.stack(self.image_list, dim=0).to(device)
        self.masked_img = torch.stack(self.masked_img, dim=0).to(device)
        self.meta_feature = torch.stack(self.meta_feature, dim=0).to(device)
        self.length = len(self.image_list)

        print(f"Maksed data: {self.masked_img.shape}")
        print(f"Raw Image data: {self.image_tensor.shape}")

    def __getitem__(self, index):
        tensor_img = self.image_tensor[index]
        masked_img = self.masked_img[index]
        meta_feature = self.meta_feature[index]
        return tensor_img, masked_img, meta_feature

    def __len__(self):
        return self.length
"""   

class ad_ris_dataset(data.Dataset):
    def __init__(self, data_path_dict: dict, path_name: str, model, preprocess, tokenizer, 
                 device, object_names, target_objects_list, sam_text_list, sam_num_list, overlap_count, kernel_size, stride, save=False, save_path=None):
        #sam_text_list와 sam_num_list는 object_names와 같은 순서로 주어지는 SAM에 사용하기 위한 정보들
        #Example: Juice bottle
        #object_names = ["a banana picture label", "a 100% text label",  "a bottle cap", "a black background"]
        #sam_text_list= ["a fruit picture label", "a 100% text label",  "a bottle cap"],
        #sam_num_list = [1, 1, 1]
        #target_objects_list = [
        #    ["a banana picture label"],
        #    ["a banana picture label", "a 100% text label"],
        #    ["a banana picture label"],
        #    ["a 100% text label"],
        #    ["a banana picture label", "a 100% text label"]
        #]

        os.makedirs(save_path, exist_ok=True)

        self.image_list = []
        self.img_paths = []
        self.masked_img = []
        model = model.to(device)

        self.sam_mask_list = []

        self.object_names = object_names
        self.meta_feature = []
        sam_info_check(object_names, sam_text_list, target_objects_list)

        for img_path in data_path_dict[path_name]:
            object_names = self.object_names.copy()
            rule_mask_list = []
            temp_meta_feature = []
            
            temp_img = Image.open(img_path)
            padded_temp_img = zero_max_pad(temp_img)
            print(img_path, padded_temp_img.size)
            
            self.image_list.append(preprocess(padded_temp_img))

            
            #masked_imgs는 clip segmentation을 통해 object마다 대략적인 segmentation을 한 결과, object_names의 길이 만큼 마스크가 들어있음.
            #_, masked_imgs = generate_refer_masked_region_for_sam(model, preprocess, tokenizer, object_names, target_objects_list, 
            #                                            temp_img, device, overlap_count=overlap_count, kernel_size=kernel_size, stride = stride, 
            #                                            draw_option=True)
            masked_imgs = [temp_img for _ in range(len(object_names))]

            l1 = find_unique_strings(target_objects_list)
            remove_target_obj_idx_list = []
            remove_target_obj_name_list = []
            print(object_names)

            for obj_idx, obj_name in enumerate(object_names):
                if obj_name not in l1:
                    remove_target_obj_idx_list.append(obj_idx)  
                    remove_target_obj_name_list.append(obj_name)
            print(remove_target_obj_name_list)
            print(remove_target_obj_idx_list)

            masked_images_for_sam = []
            for obj_name in remove_target_obj_name_list:
                object_names.remove(obj_name)
            
            for mask_idx, masked_img in enumerate(masked_imgs):
                if mask_idx not in remove_target_obj_idx_list:
                    masked_images_for_sam.append(masked_img)
            masked_imgs = masked_images_for_sam
            
            masked_imgs = [zero_max_pad(masked_img) for masked_img in masked_imgs]
            
            
            if save:
                img_name = os.path.basename(img_path).split(".")[0]
                save_img_dir = os.path.join(save_path, img_name)

                if not os.path.exists(save_img_dir):
                    os.mkdir(save_img_dir)

                padded_temp_img.save(os.path.join(save_img_dir, "origin.png"))

                for mask_idx, masked_img in enumerate(masked_imgs):
                    mask_img_name = f"clip_image_with_{object_names[mask_idx]}_mask_{mask_idx}.png"
                    masked_img.save(os.path.join(save_img_dir, mask_img_name))
            
            img_name = os.path.basename(img_path).split(".")[0]
            save_img_dir = os.path.join(save_path, img_name)
            if not os.path.exists(save_img_dir):
                os.mkdir(save_img_dir)
            padded_temp_img.save(os.path.join(save_img_dir, "origin.png"))
            sam_mask_list = []

            with torch.no_grad():
                torch.cuda.empty_cache()
                #Load CLIP and SAM model for Referring Image Segmentation
                #clip_model, sam_model = get_ris_models(device)

                for sam_idx, masked_image_for_sam in enumerate(masked_imgs):
                    sam_mask = run_ris(clip_model, sam_model, raw_image=padded_temp_img, masked_images=[masked_image_for_sam], 
                                    text_list=[sam_text_list[sam_idx]], num_object_list=[sam_num_list[sam_idx]], device="cuda")
                    sam_mask_list.append(sam_mask)
                    print("mask shape: ", sam_mask.shape)
            

            rule_sam_mask_list = [ [] for _ in range(len(target_objects_list))]
            rule_sam_masked_image_list = []
            for rule_idx, target_list in enumerate(target_objects_list):
                for target in target_list:
                    mask_idx = object_names.index(target)
                    selected_mask = sam_mask_list[mask_idx]
                    rule_sam_mask_list[rule_idx].append(selected_mask)
                rule_sam_mask_list[rule_idx] = combine_masks(rule_sam_mask_list[rule_idx])
                masked_img = apply_bi_mask_to_image(padded_temp_img, rule_sam_mask_list[rule_idx], display_option=False)
                rule_sam_masked_image_list.append(masked_img)

                meta_feature = extract_feature_from_masked_image(masked_img)
                temp_meta_feature.append(meta_feature)
            
            self.sam_mask_list.append(rule_sam_mask_list)

            if save:
                img_name = os.path.basename(img_path).split(".")[0]
                save_img_dir = os.path.join(save_path, img_name)

                if not os.path.exists(save_img_dir):
                    os.mkdir(save_img_dir)

                padded_temp_img.save(os.path.join(save_img_dir, "origin.png"))

                for mask_idx, masked_img in enumerate(rule_sam_masked_image_list):
                    mask_img_name = f"ris_image_with_{target_objects_list[mask_idx]}_mask_{mask_idx}.png"
                    masked_img.save(os.path.join(save_img_dir, mask_img_name))

            masked_imgs = [preprocess(masked_img) for masked_img in rule_sam_masked_image_list]
            masked_imgs = torch.stack(masked_imgs, dim=0)
            self.masked_img.append(masked_imgs)
            self.img_paths.append(img_path)

            temp_meta_feature = torch.stack(temp_meta_feature, dim=0)
            self.meta_feature.append(temp_meta_feature)

        self.image_tensor = torch.stack(self.image_list, dim=0).to(device)
        self.masked_img = torch.stack(self.masked_img, dim=0).to(device)
        self.meta_feature = torch.stack(self.meta_feature, dim=0).to(device)
        self.length = len(self.image_list)

        print(f"Maksed data: {self.masked_img.shape}")
        print(f"Raw Image data: {self.image_tensor.shape}")

    def __getitem__(self, index):
        tensor_img = self.image_tensor[index]
        masked_img = self.masked_img[index]
        meta_feature = self.meta_feature[index]
        return tensor_img, masked_img, meta_feature

    def __len__(self):
        return self.length 
    

class ad_lisa_dataset(data.Dataset):
    def __init__(self, img_path_list, image_list, lisa_masked_image_list, target_objects_list, save, save_path, device):
        #print(data_path_dict[path_name])
        #self.image_list = [preprocess(Image.open(img_path).resize((224,224))) for img_path in data_path_dict[path_name]]
        #self.image_tensor = torch.stack(self.image_list, dim=0).to(device)
        #self.length = len(self.image_list)
        _, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        

        self.image_list = []
        self.img_paths = []
        self.masked_img = []
        print(device)

        for img_idx, temp_img in enumerate(image_list):
            padded_temp_img = zero_max_pad(temp_img)
            
            self.image_list.append(clip_preprocess(padded_temp_img))

            masked_imgs = lisa_masked_image_list[img_idx]
            masked_imgs = [zero_max_pad(masked_img) for masked_img in masked_imgs]

            img_path = img_path_list[img_idx]
            if save:
                img_name = os.path.basename(img_path).split(".")[0]
                save_img_dir = os.path.join(save_path, img_name)

                if not os.path.exists(save_img_dir):
                    os.mkdir(save_img_dir)

                padded_temp_img.save(os.path.join(save_img_dir, "origin.png"))

                for mask_idx, masked_img in enumerate(masked_imgs):
                    mask_img_name = f"image_with_{target_objects_list[mask_idx]}_mask_{mask_idx}.png"
                    masked_img.save(os.path.join(save_img_dir, mask_img_name))
            
            masked_imgs = [clip_preprocess(masked_img) for masked_img in masked_imgs]
            masked_imgs = torch.stack(masked_imgs, dim=0)
            self.masked_img.append(masked_imgs)

        self.image_tensor = torch.stack(self.image_list, dim=0).to(device)
        self.masked_img = torch.stack(self.masked_img, dim=0).to(device)
        self.length = len(self.image_list)

        print(f"Maksed data: {self.masked_img.shape}")
        print(f"Raw Image data: {self.image_tensor.shape}")



    def __getitem__(self, index):
        tensor_img = self.image_tensor[index]
        masked_img = self.masked_img[index]
        return tensor_img, masked_img

    def __len__(self):
        return self.length