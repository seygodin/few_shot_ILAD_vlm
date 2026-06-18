"""
VLM zero-shot baseline evaluation on MVTec LOCO AD.
Computes classification accuracy (normal vs. anomalous) to compare with
the GPT-4o result in Table chatgpt_for_lad_1 of the paper.
"""
import argparse
import os
import json
import sys
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm


DATASETS = {
    'breakfast': 'Breakfast box',
    'splicing':  'Splicing connectors',
    'juice_bot': 'Juice bottle',
    'pushpins':  'Pushpins',
    'screw_bag': 'Screw bag',
}

RULES = {
    'breakfast': [
        'There must be two mandarins on the left',
        'There must be a peach on the left',
        'There must be oat cereal on the top right',
        'There must be almonds down the right',
        'There must be banana chips down the right',
        'Each content should not overflow.',
        'The amount of banana chips and almonds should be the same.',
    ],
    'splicing': [
        'There must be two splicing connectors.',
        'The heights of the two connectors should be the same.',
        'Only one cable must be connected',
        'Two block connector must have yellow cable.',
        'Three block connector must have blue cable.',
        'five block connector must have orange cable.',
    ],
    'juice_bot': [
        'banana juice bottle must have banana picture label.',
        'banana juice bottle must have white color juice.',
        'cherry juice bottle must have cherry picture label.',
        'cherry juice bottle must have red color juice.',
        'orange juice bottle must have orange picture label.',
        'orange juice bottle must have yellow color juice.',
        'picture label must be center.',
        'text label must be bottom.',
    ],
    'pushpins': [
        'There must be fifteen pushpins',
        'Each pushpins must be seperated by plastic case',
        'There must be only one pushpin in one part',
        'There must be no blank',
    ],
    'screw_bag': [
        'There must be two bolts',
        'There must be short bolt',
        'There must be long bolt',
        'There must be two hexagonal nuts',
        'There must be two round washers',
    ],
}

DATA_ROOT = '/data3/seungeon/data/image/mvtec_loco_anomaly_detection'

CATEGORY_DIRS = {
    'breakfast': 'breakfast_box',
    'splicing':  'splicing_connectors',
    'juice_bot': 'juice_bottle',
    'pushpins':  'pushpins',
    'screw_bag': 'screw_bag',
}


def build_prompt(ds_key: str, product_name: str) -> str:
    rules = RULES[ds_key]
    rule_text = '\n'.join(f'{i+1}. {r}' for i, r in enumerate(rules))
    return (
        f'You are an industrial quality inspection expert. '
        f'Below are rules that define a NORMAL "{product_name}" product. '
        f'Examine the image and decide whether the product is NORMAL or ANOMALOUS.\n\n'
        f'Rules for a normal {product_name}:\n{rule_text}\n\n'
        f'Based on these rules, is the product in the image NORMAL or ANOMALOUS? '
        f'Answer with exactly one word: either "normal" or "anomalous".'
    )


def parse_answer(text: str) -> str:
    text = text.strip().lower()
    if text.startswith('normal'):
        return 'normal'
    if text.startswith('anomalous') or text.startswith('anomaly'):
        return 'anomalous'
    # fallback: search in text
    if 'anomalous' in text or 'anomaly' in text or 'defect' in text or 'abnormal' in text:
        return 'anomalous'
    if 'normal' in text:
        return 'normal'
    return 'unknown'


def get_test_images(ds_key: str):
    cat_dir = os.path.join(DATA_ROOT, CATEGORY_DIRS[ds_key])
    good_dir = os.path.join(cat_dir, 'test', 'good')
    la_dir   = os.path.join(cat_dir, 'test', 'logical_anomalies')

    images = []
    if os.path.exists(good_dir):
        for f in sorted(os.listdir(good_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                images.append((os.path.join(good_dir, f), 'normal'))

    if os.path.exists(la_dir):
        for sub in sorted(os.listdir(la_dir)):
            sub_path = os.path.join(la_dir, sub)
            if os.path.isdir(sub_path):
                for f in sorted(os.listdir(sub_path)):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        images.append((os.path.join(sub_path, f), 'anomalous'))
            elif sub.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                images.append((sub_path, 'anomalous'))
    return images


def evaluate_qwen(gpu_id: int, results_path: str, model_size: str = '7B'):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    device = f'cuda:{gpu_id}'
    model_id = f'Qwen/Qwen2.5-VL-{model_size}-Instruct'
    print(f'[Qwen2.5-VL-{model_size}] Loading on {device}...', flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    all_results = {}
    for ds_key, prod_name in DATASETS.items():
        prompt_text = build_prompt(ds_key, prod_name)
        images = get_test_images(ds_key)
        print(f'  {prod_name}: {len(images)} images', flush=True)

        correct = 0
        n_unknown = 0
        for img_path, gt in tqdm(images, desc=prod_name, leave=False):
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': img_path},
                        {'type': 'text', 'text': prompt_text},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, return_tensors='pt', padding=True
            ).to(device)

            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=16)
            trimmed = out_ids[0][inputs.input_ids.shape[1]:]
            answer_text = processor.decode(trimmed, skip_special_tokens=True)
            pred = parse_answer(answer_text)
            if pred == 'unknown':
                n_unknown += 1
            if pred == gt:
                correct += 1

        acc = correct / len(images) * 100
        all_results[ds_key] = {'name': prod_name, 'accuracy': acc, 'n': len(images), 'unknown': n_unknown}
        print(f'  {prod_name}: {acc:.1f}% ({correct}/{len(images)}, unknown={n_unknown})', flush=True)

    avg = sum(v['accuracy'] for v in all_results.values()) / len(all_results)
    all_results['average'] = avg
    print(f'\nQwen2.5-VL-{model_size} average accuracy: {avg:.1f}%', flush=True)

    with open(results_path, 'w') as f:
        json.dump({'model': f'Qwen2.5-VL-{model_size}', 'timestamp': datetime.now().isoformat(), 'results': all_results}, f, indent=2)
    print(f'Saved to {results_path}', flush=True)
    return all_results


def evaluate_qwen2vl(gpu_id: int, results_path: str, model_size: str = '7B'):
    """Qwen2-VL (the previous generation) for a generational comparison vs Qwen2.5-VL."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    device = f'cuda:{gpu_id}'
    model_id = f'Qwen/Qwen2-VL-{model_size}-Instruct'
    print(f'[Qwen2-VL-{model_size}] Loading on {device}...', flush=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    all_results = {}
    for ds_key, prod_name in DATASETS.items():
        prompt_text = build_prompt(ds_key, prod_name)
        images = get_test_images(ds_key)
        print(f'  {prod_name}: {len(images)} images', flush=True)

        correct = 0
        n_unknown = 0
        for img_path, gt in tqdm(images, desc=prod_name, leave=False):
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': img_path},
                        {'type': 'text', 'text': prompt_text},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, return_tensors='pt', padding=True
            ).to(device)

            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=16)
            trimmed = out_ids[0][inputs.input_ids.shape[1]:]
            answer_text = processor.decode(trimmed, skip_special_tokens=True)
            pred = parse_answer(answer_text)
            if pred == 'unknown':
                n_unknown += 1
            if pred == gt:
                correct += 1

        acc = correct / len(images) * 100
        all_results[ds_key] = {'name': prod_name, 'accuracy': acc, 'n': len(images), 'unknown': n_unknown}
        print(f'  {prod_name}: {acc:.1f}% ({correct}/{len(images)}, unknown={n_unknown})', flush=True)

    avg = sum(v['accuracy'] for v in all_results.values()) / len(all_results)
    all_results['average'] = avg
    print(f'\nQwen2-VL-{model_size} average accuracy: {avg:.1f}%', flush=True)

    with open(results_path, 'w') as f:
        json.dump({'model': f'Qwen2-VL-{model_size}', 'timestamp': datetime.now().isoformat(), 'results': all_results}, f, indent=2)
    print(f'Saved to {results_path}', flush=True)
    return all_results


def evaluate_llava(gpu_id: int, results_path: str):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

    device = f'cuda:{gpu_id}'
    model_id = 'llava-hf/llava-v1.6-mistral-7b-hf'
    print(f'[LLaVA-1.6-Mistral-7B] Loading on {device}...', flush=True)

    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device
    )
    model.eval()

    all_results = {}
    for ds_key, prod_name in DATASETS.items():
        prompt_text = build_prompt(ds_key, prod_name)
        images = get_test_images(ds_key)
        print(f'  {prod_name}: {len(images)} images', flush=True)

        correct = 0
        n_unknown = 0
        for img_path, gt in tqdm(images, desc=prod_name, leave=False):
            image = Image.open(img_path).convert('RGB')
            conversation = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': prompt_text},
                    ],
                }
            ]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=text, images=image, return_tensors='pt').to(device)

            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=16)
            answer_text = processor.decode(out_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            pred = parse_answer(answer_text)
            if pred == 'unknown':
                n_unknown += 1
            if pred == gt:
                correct += 1

        acc = correct / len(images) * 100
        all_results[ds_key] = {'name': prod_name, 'accuracy': acc, 'n': len(images), 'unknown': n_unknown}
        print(f'  {prod_name}: {acc:.1f}% ({correct}/{len(images)}, unknown={n_unknown})', flush=True)

    avg = sum(v['accuracy'] for v in all_results.values()) / len(all_results)
    all_results['average'] = avg
    print(f'\nLLaVA-1.6-Mistral-7B average accuracy: {avg:.1f}%', flush=True)

    with open(results_path, 'w') as f:
        json.dump({'model': 'LLaVA-1.6-Mistral-7B', 'timestamp': datetime.now().isoformat(), 'results': all_results}, f, indent=2)
    print(f'Saved to {results_path}', flush=True)
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'qwen2vl', 'llava'], required=True)
    parser.add_argument('--model_size', type=str, default='7B', help='Qwen model size, e.g. 3B or 7B')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        if args.model == 'qwen':
            suffix = f'qwen25_{args.model_size}'
        elif args.model == 'qwen2vl':
            suffix = f'qwen2vl_{args.model_size}'
        else:
            suffix = args.model
        args.output = f'results/vlm_eval_{suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    os.makedirs('results', exist_ok=True)

    if args.model == 'qwen':
        evaluate_qwen(args.gpu, args.output, model_size=args.model_size)
    elif args.model == 'qwen2vl':
        evaluate_qwen2vl(args.gpu, args.output, model_size=args.model_size)
    else:
        evaluate_llava(args.gpu, args.output)
