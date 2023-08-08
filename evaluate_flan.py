import argparse
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5TokenizerFast

from mmlu.dataset import CHOICES
from mmlu.evaluation import predict_dataset, evaluate_results


class FlanPredictor(Callable[[str], str]):

    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5TokenizerFast) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._model.eval()
        self._choice_tokens = [self._tokenizer(c).input_ids[0] for c in CHOICES]
        self._choice_tokens = torch.tensor(self._choice_tokens).long()

    def __call__(self, prompt: str) -> str:
        input_ids = self._tokenizer(prompt, return_tensors='pt').input_ids
        decoder_input_ids = self._tokenizer('', return_tensors='pt').input_ids
        decoder_input_ids = self._model._shift_right(decoder_input_ids)
        logits = self._model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids
        ).logits.flatten()
        logits = logits[self._choice_tokens]
        pred_index = torch.argmax(logits)
        pred = CHOICES[int(pred_index)]
        return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--result_dir', type=str, default='results/flan_small_new')
    parser.add_argument('--k_shot', type=int, default=0, help='The number of few-shot examples in the prompt.')
    parser.add_argument('--engine', type=str, default='google/flan-t5-small')
    args = parser.parse_args()
    print(args)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.engine)
    tokenizer = AutoTokenizer.from_pretrained(args.engine)
    predict_function = FlanPredictor(model, tokenizer)
    token_counter = lambda x: tokenizer(x, return_tensors='pt').input_ids.shape[-1]

    predict_dataset(data_dir=Path(args.data_dir),
                    result_dir=Path(args.result_dir),
                    predict_function=predict_function,
                    k_shot=args.k_shot,
                    n_workers=0,
                    timeout_s=0,
                    retries=0,
                    token_counter=token_counter,
                    max_tokens=2048)

    evaluate_results(result_dir=Path(args.result_dir))
