import argparse
import random
from pathlib import Path

from mmlu.dataset import CHOICES
from mmlu.evaluation import predict_dataset, evaluate_results


def predict_random(prompt: str) -> str:
    return random.choice(CHOICES)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_de')
    parser.add_argument('--result_dir', type=str, default='results/random')
    parser.add_argument('--k_shot', type=int, default=0, help='The number of few-shot examples in the prompt.')
    args = parser.parse_args()
    print(args)

    predict_dataset(data_dir=Path(args.data_dir),
                    result_dir=Path(args.result_dir),
                    predict_function=predict_random,
                    k_shot=args.k_shot)

    evaluate_results(result_dir=Path(args.result_dir))
