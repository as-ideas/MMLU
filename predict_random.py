import random
import time
from pathlib import Path

from prediction import LLMPredictor, predict_dataset


class RandomPredictor(LLMPredictor):

    def __call__(self, prompt: str):
        time.sleep(1)
        return random.choice(['A', 'B', 'C', 'D'])


if __name__ == '__main__':
    data_dir = Path('/Users/cschaefe/datasets/nlp/mmlu/data')
    result_dir = Path(f'results/random_0_shot')
    predictor = RandomPredictor()
    predict_dataset(data_dir=data_dir,
                    result_dir=result_dir,
                    predictor=predictor,
                    k_shot=0,
                    n_workers=10)
