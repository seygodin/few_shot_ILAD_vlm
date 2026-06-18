"""
GPT-4o evaluation on MVTec LOCO AD with SIMPLE vs CHAIN-OF-THOUGHT prompts.
Addresses Reviewer #2 concern (R2-7): is the GPT-4o comparison unfair because the
prompt is too simple? We re-run GPT-4o with an improved step-by-step (CoT) prompt
on the IDENTICAL image subset and compare accuracy, to test whether better
prompting closes the gap to our few-shot method.

Same good-vs-logical protocol, same RULES, and same parse as eval_vlm_baseline.py
for consistency with the paper's VLM table.
"""
import argparse
import base64
import io
import json
import os
import time
from datetime import datetime

import requests
from PIL import Image

from eval_vlm_baseline import DATASETS, RULES, get_test_images, parse_answer

OPENAI_URL = 'https://api.openai.com/v1/chat/completions'


def build_prompt_simple(ds_key, product_name):
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


def build_prompt_cot(ds_key, product_name):
    rules = RULES[ds_key]
    rule_text = '\n'.join(f'{i+1}. {r}' for i, r in enumerate(rules))
    return (
        f'You are an industrial quality inspection expert. '
        f'Below are rules that define a NORMAL "{product_name}" product. '
        f'Examine the image VERY carefully and reason step by step, checking each rule '
        f'one at a time against exactly what you observe in the image (including object '
        f'counts, colors, positions, and arrangement).\n\n'
        f'Rules for a normal {product_name}:\n{rule_text}\n\n'
        f'For each numbered rule, state whether the image satisfies it and briefly why. '
        f'Pay special attention to counting and spatial arrangement. '
        f'After checking all rules, output your final verdict on a new line in EXACTLY this '
        f'format: "FINAL: normal" if every rule is satisfied, otherwise "FINAL: anomalous".'
    )


def parse_cot(text):
    # prefer the explicit FINAL: line
    for line in reversed(text.strip().splitlines()):
        ls = line.strip().lower()
        if 'final' in ls:
            if 'anomal' in ls:
                return 'anomalous'
            if 'normal' in ls:
                return 'normal'
    return parse_answer(text)


def encode_image(path, max_side=768):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def call_gpt4o(api_key, model, prompt, b64, cot):
    payload = {
        'model': model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
            ],
        }],
        'max_tokens': 600 if cot else 16,
        'temperature': 0.0,
    }
    r = requests.post(OPENAI_URL,
                      headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                      json=payload, timeout=90)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['simple', 'cot'], required=True)
    ap.add_argument('--model', default='gpt-4o')
    ap.add_argument('--cap', type=int, default=0, help='max good/logical images per category; <=0 means ALL (full test set)')
    ap.add_argument('--workers', type=int, default=8, help='concurrent API requests')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    api_key = os.environ['OPENAI_API_KEY']
    cot = args.mode == 'cot'
    builder = build_prompt_cot if cot else build_prompt_simple
    parser_fn = parse_cot if cot else parse_answer

    out = args.output or f'results/vlm_eval_gpt4o_{args.mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    os.makedirs('results', exist_ok=True)

    def score_one(item, prompt):
        img_path, gt = item
        b64 = encode_image(img_path)
        txt = ''
        for attempt in range(5):
            try:
                txt = call_gpt4o(api_key, args.model, prompt, b64, cot)
                break
            except Exception as e:
                if attempt == 4:
                    print(f'    FAILED {img_path}: {e}', flush=True)
                else:
                    time.sleep(2 ** attempt)
        return parser_fn(txt), gt

    all_results = {}
    for ds_key, prod_name in DATASETS.items():
        images = get_test_images(ds_key)
        good = [x for x in images if x[1] == 'normal']
        anom = [x for x in images if x[1] == 'anomalous']
        if args.cap > 0:            # cap<=0 => use ALL images (full test set, matches paper protocol)
            good, anom = good[:args.cap], anom[:args.cap]
        subset = good + anom
        prompt = builder(ds_key, prod_name)
        print(f'[{args.mode}] {prod_name}: {len(good)} good + {len(anom)} logical = {len(subset)} (workers={args.workers})', flush=True)

        correct = 0
        n_unknown = 0
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for pred, gt in ex.map(lambda it: score_one(it, prompt), subset):
                if pred == 'unknown':
                    n_unknown += 1
                if pred == gt:
                    correct += 1
        acc = correct / len(subset) * 100 if subset else 0.0
        all_results[ds_key] = {'name': prod_name, 'accuracy': acc, 'n': len(subset),
                               'n_good': len(good), 'n_anom': len(anom), 'unknown': n_unknown}
        print(f'  {prod_name}: {acc:.1f}% ({correct}/{len(subset)}, unknown={n_unknown})', flush=True)

    avg = sum(v['accuracy'] for v in all_results.values()) / len(all_results)
    all_results['average'] = avg
    print(f'\nGPT-4o ({args.mode}) average accuracy: {avg:.1f}%', flush=True)
    with open(out, 'w') as f:
        json.dump({'model': f'GPT-4o-{args.mode}', 'cap': args.cap,
                   'timestamp': datetime.now().isoformat(), 'results': all_results}, f, indent=2)
    print(f'Saved to {out}', flush=True)


if __name__ == '__main__':
    main()
