from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from tqdm import tqdm

from mmlu.dataset import get_subjects, Dataset, gen_prompt, read_or_create_result_df, file_to_subject
from mmlu.threading_utils import PredictionWithTimeout, PredictionWorker


def predict_dataset(data_dir: Path,
                    result_dir: Path,
                    predict_function: Callable[[str], str],
                    subjects: Optional[List[str]] = None,
                    k_shot: int = 0,
                    n_workers: int = 0,
                    timeout_s: int = 50,
                    retries: int = 3):

    result_dir.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        subjects = get_subjects(data_dir)

    for subject_index, subject in enumerate(subjects):
        print(f'--------------------\nPredict {subject_index}/{len(subjects)}: {subject}')
        dataset = Dataset.from_dir(data_dir=data_dir, subject=subject)
        result_file = result_dir / f'{subject}_result.csv'
        result_df = read_or_create_result_df(result_file, dataset)
        prompts = [gen_prompt(dataset, i, k_shot=k_shot) for i in range(len(dataset))]
        prompt_jobs = [(p, i) for i, p in enumerate(prompts) if len(result_df.loc[i, 'prediction']) != 1]

        if timeout_s > 0 and retries > 0:
            predict_function = PredictionWithTimeout(func=predict_function,
                                                     timeout_s=timeout_s,
                                                     retries=retries)

        if len(prompt_jobs) > 0 and n_workers > 0:
            prediction_worker = PredictionWorker(predict_function)
            pool = Pool(processes=n_workers)
            for pred, index in tqdm(pool.imap_unordered(prediction_worker, prompt_jobs),
                                    total=len(prompt_jobs)):
                result_df.loc[index, 'prediction'] = pred
            pool.terminate()
        elif len(prompt_jobs) > 0 and n_workers == 0:
            for prompt, index in tqdm(prompt_jobs, total=len(prompt_jobs)):
                pred = predict_function(prompt)
                result_df.loc[index, 'prediction'] = pred

        result_df.to_csv(result_file, sep=',', encoding='utf-8', index=False)
        tp, pred = sum(get_true_pos(result_df)), len(get_pred(result_df))
        print(f'Correct: {tp}/{pred}, accuracy: {get_accuracy(result_df):#.3}')


def evaluate_results(result_dir: Path,
                     subjects: Optional[List[str]] = None,
                     out_file: Optional[Path] = None) -> None:

    result_files = result_dir.glob('**/*.csv')
    subject_to_file = {file_to_subject(f): f for f in result_files}
    if subjects is not None:
        subject_to_file = {s: f for s, f in subject_to_file.items() if s in subjects}

    print('--------------------------\nAccuracy by subject:')
    out_rows = []
    for subject in sorted(subject_to_file.keys()):
        result_df = pd.read_csv(subject_to_file[subject], sep=',', encoding='utf-8')
        true_pos = get_true_pos(result_df)
        num_labels = len(get_pred(result_df))
        acc = sum(true_pos) / max(num_labels, 1)
        out_rows.append({'subject': subject, 'true_pos': sum(true_pos),
                         'num_labels': num_labels, 'accuracy': acc})
        print(f'{subject}: {acc:#.3}')

    sum_true_pos = sum(row['true_pos'] for row in out_rows)
    sum_labels = sum(row['num_labels'] for row in out_rows)
    micro_avg_acc = sum_true_pos / sum_labels

    print(f'---------------------\nMicro-averaged accuracy: {micro_avg_acc:#.2}')

    if out_file is not None:
        out_df = pd.DataFrame(out_rows)
        out_df.to_csv(out_file, sep=',', encoding='utf-8')


def get_pred(result_df: pd.DataFrame) -> List[bool]:
    return [p for p in result_df['prediction'] if len(str(p)) == 1]


def get_true_pos(result_df: pd.DataFrame) -> List[bool]:
    return (result_df['prediction'] == result_df['label']).tolist()


def get_accuracy(result_df: pd.DataFrame) -> float:
    return sum(result_df['prediction'] == result_df['label']) / len(result_df)