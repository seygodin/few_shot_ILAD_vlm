import torch
import open_clip
from PIL import Image
import sys
from time import time
from memory_profiler import memory_usage

start_mem = memory_usage()[0]
model_name = "ViT-B-32"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s34b_b79k')
model.eval().cuda()

tokenizer = open_clip.get_tokenizer(model_name)
num_detection = 1
num_iter = 1
img_path = "/data/seungeon/orig/screw_bag/train/good/000.png"

time_list = []

medium_mem = memory_usage()[0]

for _ in range(num_iter):
    torch.cuda.synchronize()
    start_time = time()
    raw_image = Image.open(img_path)

    draw_image = raw_image.copy()
    width, height = raw_image.size
    kernel_size = 250
    stride = 50
    device = "cuda"
    num_x = len(range(0, width-kernel_size+1, stride))
    num_y = len(range(0,height-kernel_size+1, stride))

    object_names = ["an object", "an object", "an object"]
    object_texts = tokenizer(object_names)
    object_texts = object_texts.to(device)

    #sample_image  = preprocess(raw_image.crop((0, 0, 250, 250))).unsqueeze(0).to(device)
    #sample_text = object_texts[0:1]

    #jit_encode_image = torch.jit.trace_module(model, {"encode_image": sample_image.cuda()})
    #jit_encode_text = torch.jit.trace_module(model, {"encode_text": sample_text.cuda()})

    jit_encode_image = model
    jit_encode_text = model

    
    for idx in range(num_detection):
        cropped_image_list = []
        

        for x_idx, left_x in enumerate(range(0, width-kernel_size+1, stride)):
            for y_idx, left_y in enumerate(range(0,height-kernel_size+1, stride)):  
                right_x = left_x + kernel_size
                right_y = left_y + kernel_size
                cropped_image = raw_image.crop((left_x, left_y, right_x, right_y))
                
                cropped_image = preprocess(cropped_image).unsqueeze(0)
                cropped_image_feature = jit_encode_image.encode_image(cropped_image.cuda())
                #cropped_image_list.append(cropped_image)

        #cropped_image_features = torch.stack(cropped_image_list).to(device)
        #print(cropped_image_features.shape)
        #print(object_texts.shape)
        #with torch.no_grad():
        #    image_features = jit_encode_image.encode_image(cropped_image_features)
        #    text_features = jit_encode_text.encode_text(object_texts)
        #print(image_features.shape, text_features.shape)
    
    detection_mem = memory_usage()[0]

        #time_list.append(end_time-start_time)

    #time_list = time_list[1:]
    #print("Time consumption")
    #print(sum(time_list)/  len(time_list))
    #sys.exit()

    raw_image  = preprocess(Image.open(img_path)).unsqueeze(0).cuda()
    masked_image  = preprocess(Image.open(img_path)).unsqueeze(0).cuda()
    masked_images = [preprocess(Image.open(img_path)) for _ in range(100)]
    batch_image = torch.stack(masked_images)
    batch_image = batch_image.cuda()


    total_loss = 0
    total_neg_loss = 0
    global_image_feature = model.encode_image(raw_image)

    for _ in range(5):
        masked_image_feature = model.encode_image(batch_image)
        masked_image_feature = model.encode_image(batch_image)
        image_feature = (global_image_feature + masked_image_feature)

        global_text_feature = model.encode_text(object_texts)
        global_text_feature = model.encode_text(object_texts)
        text_feature = (global_text_feature + global_text_feature)

    prob= image_feature @ text_feature.T
    end_time = time()
    torch.cuda.synchronize()
    #print(end_time-start_time)
    time_list.append(end_time-start_time)

end_mem = memory_usage()[0]

#time_list = time_list[1:]
print("Time consumption")
print(sum(time_list)/len(time_list))

print("Used memory")
print(f"Model: {medium_mem-start_mem} MB")
print(f"Detection: {detection_mem-medium_mem}")
print(f"Inference: {end_mem-detection_mem} MB")
print(f"Full: {end_mem-start_mem} MB")