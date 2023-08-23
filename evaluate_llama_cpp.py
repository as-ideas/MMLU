import json
import argparse
from pathlib import Path
from typing import Callable

import requests

from mmlu.evaluation import predict_dataset, evaluate_results

URL_TOKENIZE = 'http://localhost:8080/tokenize'
URL_PREDICT = 'http://localhost:8080/completion'
HEADERS = {'Content-Type': 'application/json'}


class LLamaCPPPredictor(Callable[[str], str]):

    def __init__(self) -> None:
        choice_inds = set()
        for t in ['A', 'B', 'C', 'D', ' A', ' B', ' C', ' D']:
            response = requests.post(URL_TOKENIZE, headers=HEADERS, json={'content': t})
            token_ind = int(json.loads(response.text)['tokens'][-1])
            choice_inds.add(token_ind)
        self._logit_bias = [[ind, 200] for ind in choice_inds]

    def __call__(self, prompt: str) -> str:
        data = {
            'prompt': prompt,
            'n_predict': 1,
            'logit_bias': self._logit_bias
        }
        response = requests.post(URL_PREDICT, headers=HEADERS, json=data)
        out = json.loads(response.text)['content'][-1]
        print(out)
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--result_dir', type=str, default='results/llama_cpp')
    parser.add_argument('--k_shot', type=int, default=0, help='The number of few-shot examples in the prompt.')
    args = parser.parse_args()
    print(args)

    prediction_function = LLamaCPPPredictor()

    def token_counter(prompt: str) -> int:
        response = requests.post(URL_TOKENIZE, headers=HEADERS, json={'content': prompt})
        return len(json.loads(response.text)['tokens'])

    predict_dataset(data_dir=Path(args.data_dir),
                    result_dir=Path(args.result_dir),
                    predict_function=prediction_function,
                    token_counter=token_counter,
                    max_tokens=4096,
                    k_shot=args.k_shot)

    evaluate_results(result_dir=Path(args.result_dir))
