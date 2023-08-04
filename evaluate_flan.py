import os
import torch
import numpy as np
from pathlib import Path

import openai

from mmlu.evaluation import predict_dataset, evaluate_results

openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview'
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv('OPENAI_API_KEY')

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
model.eval()


def predict_flan(prompt: str) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    decoder_input_ids = tokenizer("", return_tensors="pt").input_ids
    decoder_input_ids = model._shift_right(decoder_input_ids)
    logits = model(
        input_ids=input_ids, decoder_input_ids=decoder_input_ids
    ).logits.flatten()

    probs = (
        torch.nn.functional.softmax(
            torch.tensor(
                [
                    logits[tokenizer("A").input_ids[0]],
                    logits[tokenizer("B").input_ids[0]],
                    logits[tokenizer("C").input_ids[0]],
                    logits[tokenizer("D").input_ids[0]],
                ]
            ),
            dim=0,
        )
            .detach()
            .cpu()
            .numpy()
    )
    pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
    return pred


if __name__ == '__main__':
    data_dir = Path('/Users/cschaefe/datasets/nlp/mmlu/data')
    result_dir = Path(f'results/flan_small')

    predict_dataset(data_dir=data_dir,
                    result_dir=result_dir,
                    predict_function=predict_flan,
                    subjects=['human_sexuality'],
                    k_shot=0,
                    n_workers=0,
                    timeout_s=0,
                    retries=0)

    evaluate_results(result_dir=result_dir, out_file=Path('/tmp/chatgpt.csv'))
