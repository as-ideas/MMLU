import os
from pathlib import Path
from typing import Callable

import openai
import argparse

from mmlu.evaluation import predict_dataset, evaluate_results

openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview'
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv('OPENAI_API_KEY')


class AzurePredictor(Callable[[str], str]):

    def __init__(self, engine: str) -> None:
        self._engine = engine

    def __call__(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            engine=self._engine,
            messages=[
                {'role': 'system', 'content': ''},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0,
            max_tokens=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            # Increase likelihood of tokens: [' A', ' B', ' C', ' D', 'A', 'B', 'C', 'D']
            logit_bias={362: 100, 426: 100, 356: 100, 423: 100,
                        32: 100, 33: 100, 34: 100, 35: 100},
            stop=None)
        pred = response['choices'][0]['message']['content'].replace(' ', '')
        return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--result_dir', type=str, default='results/azure')
    parser.add_argument('--engine', type=str, default='gpt-35-turbo',
                        help='The engine to use. Choices: [gpt-35-turbo, gpt-4]')
    parser.add_argument('--k_shot', type=int, default=0, help='The number of few-shot examples in the prompt.')
    parser.add_argument('--n_workers', type=int, default=2, help='The number of worker threads to use for api calls.')
    parser.add_argument('--timeout', type=float, default=60, help='The timeout for api calls in seconds.')
    parser.add_argument('--retries', type=int, default=3, help='The number of retries.')
    args = parser.parse_args()
    print(args)

    predict_function = AzurePredictor(engine=args.engine)

    predict_dataset(data_dir=Path(args.data_dir),
                    result_dir=Path(args.result_dir),
                    predict_function=predict_function,
                    k_shot=0,
                    n_workers=args.n_workers,
                    timeout_s=50,
                    retries=3)

    evaluate_results(result_dir=Path(args.result_dir))
