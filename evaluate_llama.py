import argparse
from pathlib import Path
from typing import Callable

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

from mmlu.dataset import CHOICES
from mmlu.evaluation import predict_dataset, evaluate_results
from mmlu.prediction_utils import LogitBiasProcessor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class LLama2Predictor(Callable[[str], str]):

    def __init__(self, model: LlamaForCausalLM, tokenizer: LlamaTokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._model.eval()
        self._choice_tokens = [self._tokenizer(c).input_ids[0] for c in CHOICES]
        choice_ids = [tokenizer(c).input_ids[-1] for c in CHOICES]
        logit_bias = {c_id: 200 for c_id in choice_ids}
        # Add 200 to logits for choice tokens ['A',  'B', 'C', 'D']
        self._logits_processor = LogitBiasProcessor(logit_bias=logit_bias)
        self._choice_tokens = torch.tensor(self._choice_tokens).long()

    def __call__(self, prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generated_ids = model.generate(inputs.input_ids,
                                       max_new_tokens=1,
                                       do_sample=False,
                                       logits_processor=[self._logits_processor]).to(device)
        out = tokenizer.batch_decode(generated_ids.cpu(),
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
        out = out[-1:]
        print(prompt)
        print(out)
        print('*****************')
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--result_dir', type=str, default='results/Llama-2-7b-hf-0shot')
    parser.add_argument('--k_shot', type=int, default=0, help='The number of few-shot examples in the prompt.')
    parser.add_argument('--engine', type=str, default='meta-llama/Llama-2-7b-hf')
    args = parser.parse_args()
    print(args)

    model = LlamaForCausalLM.from_pretrained(args.engine).to(device)
    tokenizer = LlamaTokenizer.from_pretrained(args.engine)

    predict_function = LLama2Predictor(model=model, tokenizer=tokenizer)

    def token_counter(prompt: str) -> int:
        return tokenizer(prompt, return_tensors='pt').input_ids.shape[-1]

    predict_dataset(data_dir=Path(args.data_dir),
                    result_dir=Path(args.result_dir),
                    predict_function=predict_function,
                    k_shot=args.k_shot,
                    n_workers=0,
                    timeout_s=0,
                    retries=0,
                    token_counter=token_counter,
                    max_tokens=4192)

    evaluate_results(result_dir=Path(args.result_dir))
