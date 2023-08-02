from abc import ABC
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

from dataset import get_subjects, Dataset, gen_prompt, read_or_create_result_df


class LLMPredictor(ABC):

    def __call__(self, prompt: str) -> str:
        """
        Predicts a response for a question prompt. The response should be a single letter as in ['A', 'B', 'C', 'D']
        Args:
            prompt: Exam question prompt.

        Returns: Response text of the model as a single letter.

        """
        raise NotImplementedError()


class PredictionWorker:

    def __init__(self, predictor: LLMPredictor) -> None:
        self.predictor = predictor

    def __call__(self, prompt_index: Tuple[str, int]) -> Tuple[str, int]:
        prompt, index = prompt_index
        pred = self.predictor(prompt)
        return pred, index


def predict_dataset(data_dir: Path,
                    result_dir: Path,
                    predictor: LLMPredictor,
                    k_shot: int = 0,
                    n_workers: int = 0):

    result_dir.mkdir(parents=True, exist_ok=True)
    subjects = get_subjects(data_dir)

    for subject_index, subject in enumerate(subjects):
        print(f'{subject_index}/{len(subjects)}: {subject}')
        dataset = Dataset.from_dir(data_dir=data_dir, subject=subject)
        result_file = result_dir / f'{subject}_result.csv'
        result_df = read_or_create_result_df(result_file, dataset)
        prompts = [gen_prompt(dataset, i, k_shot=k_shot) for i in range(len(dataset))]
        prompt_jobs = [(p, i) for i, p in enumerate(prompts) if result_df.loc[i, 'prediction'] is None]

        if n_workers > 0:
            prediction_worker = PredictionWorker(predictor)
            pool = Pool(processes=n_workers)
            for pred, index in tqdm(pool.imap_unordered(prediction_worker, prompt_jobs),
                                                   total=len(prompt_jobs)):
                result_df.loc[index, 'prediction'] = pred
        else:
            for prompt, index in tqdm(prompt_jobs, total=len(prompt_jobs)):
                pred = predictor(prompt)
                result_df.loc[index, 'prediction'] = pred

        result_df.to_csv(result_file, sep=',', encoding='utf-8')