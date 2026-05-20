import torch
from time import time



inference_time_list = []
for idx in range(50):
    print(idx)
    torch.cuda.synchronize()
    start_time = time()
    for x_idx, left_x in enumerate(range(0, width-kernel_size+1, stride)):
        for y_idx, left_y in enumerate(range(0,height-kernel_size+1, stride)):
            right_x = left_x + kernel_size
            right_y = left_y + kernel_size
            cropped_image = raw_image.crop((left_x, left_y, right_x, right_y))
            
            cropped_image = preprocess(cropped_image).unsqueeze(0)

            cropped_image = cropped_image.to(device)

            
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(cropped_image)
                text_features = model.encode_text(object_texts[0:1])
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    end_time = time()
    torch.cuda.synchronize()
    inference_time_list.append(end_time-start_time)
    continue