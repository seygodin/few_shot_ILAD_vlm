import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from lavis.models import load_model
from torch.cuda.amp import autocast


raw_image = Image.open("/data/seungeon/orig/breakfast_box/test/good/090.png")

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
model = load_model("blip2_image_text_matching", "coco")
model = model.to(device)
# model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)

caption = "merlion in Singapore"

img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
txt = text_processors["eval"](caption)  
print(img.shape)
output = model({"image": img, "text_input": txt})
print(output)