from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional
from sklearn.metrics import accuracy_score

import pandas as pd
from tqdm import tqdm

from mmlu.dataset import get_subjects, Dataset, gen_prompt, read_or_create_result_df, file_to_subject
from mmlu.threading_utils import PredictionWithTimeout, PredictionWorker


SAVE_STEPS = 50


def predict_dataset(data_dir: Path,
                    result_dir: Path,
                    predict_function: Callable[[str], str],
                    k_shot: int = 0,
                    n_workers: int = 0,
                    timeout_s: int = 50,
                    retries: int = 3,
                    subjects: Optional[List[str]] = None,
                    token_counter: Optional[Callable[[str], int]] = None,
                    max_tokens: Optional[int] = None):
    """
    Performs model predictions over the dataset and stores the result in a dataframe.

    Args:
        data_dir (Path): The directory containing the dataset files.
        result_dir (Path): The directory where prediction results will be stored.
        predict_function (Callable[[str], str]): A callable function that takes a prompt
                                                 string as input and returns the prediction as a string.
        k_shot (int): The number of examples for k-shot learning. Defaults to 0.
        n_workers (int): The number of worker processes to use for parallel prediction.
                                   If 0, prediction will be done in the main thread. Defaults to 0.
        timeout_s (int): The timeout in seconds for each prediction job. Defaults to 50.
        retries (int): The number of times to retry a timed-out prediction job. Defaults to 3.
        subjects (List[str], optional): A list of subjects from the dataset to be used for prediction.
                                        If None, all subjects in the dataset will be used. Defaults to None.
        token_counter (Callable[[str], int], optional): A callable function that takes a prompt string as
                                                        input and returns the number of tokens in it.
                                                        Defaults to None.
        max_tokens (int, optional): The maximum number of tokens allowed in a prompt. Defaults to None.

    Returns:
        None: The function doesn't return any value directly, but saves the prediction results to the result_dir.
    """

    result_dir.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        subjects = get_subjects(data_dir)

    for subject_index, subject in enumerate(subjects):
        print(f'--------------------\nPredict {subject_index}/{len(subjects)}: {subject}')
        dataset = Dataset.from_dir(data_dir=data_dir, subject=subject)
        result_file = result_dir / f'{subject}_result.csv'
        result_df = read_or_create_result_df(result_file, dataset)
        prompts = [gen_prompt(dataset=dataset, index=i, k_shot=k_shot,
                              token_counter=token_counter, max_tokens=max_tokens)
                   for i in range(len(dataset))]
        prompt_jobs = [(p, i) for i, p in enumerate(prompts)
                       if len(result_df.loc[i, 'prediction']) != 1]

        if timeout_s > 0 and retries > 0:
            predict_function = PredictionWithTimeout(func=predict_function,
                                                     timeout_s=timeout_s,
                                                     retries=retries)

        if len(prompt_jobs) > 0 and n_workers > 0:
            prediction_worker = PredictionWorker(predict_function)
            pool = Pool(processes=n_workers)
            for j, (pred, index) in enumerate(tqdm(pool.imap_unordered(prediction_worker, prompt_jobs),
                                                   total=len(prompt_jobs))):
                result_df.loc[index, 'prediction'] = pred
                if (j + 1) % SAVE_STEPS == 0:
                    result_df.to_csv(result_file, sep=',', encoding='utf-8', index=False)
            pool.terminate()
        elif len(prompt_jobs) > 0 and n_workers == 0:
            for j, (prompt, index) in enumerate(tqdm(prompt_jobs, total=len(prompt_jobs))):
                pred = predict_function(prompt)
                result_df.loc[index, 'prediction'] = pred
                if (j + 1) % SAVE_STEPS == 0:
                    result_df.to_csv(result_file, sep=',', encoding='utf-8', index=False)

        result_df.to_csv(result_file, sep=',', encoding='utf-8', index=False)
        acc = accuracy_score(y_true=result_df['label'], y_pred=result_df['prediction'])
        true_pos = sum(result_df['prediction'] == result_df['label'])
        num_labels = len(result_df['label'])

        print(f'Accuracy: {acc:#.3} ({true_pos}/{num_labels})')


def evaluate_results(result_dir: Path,
                     subjects: Optional[List[str]] = None,
                     out_file: Optional[Path] = None) -> None:
    """
    Evaluates the prediction results stored in result_dir and calculate accuracy metrics.

    Args:
        result_dir (Path): The directory containing the prediction result files in CSV format (e.g., *_result.csv).
        subjects (List[str], optional): A list of subjects for which to evaluate the results.
                                        If None, all subjects found in result_dir will be evaluated.
                                        Defaults to None.
        out_file (Path, optional): The file path where the evaluation results will be saved as a CSV file.
                                   If None, the evaluation results will only be printed on the console.
                                   Defaults to None.

    Returns:
        None: The function doesn't return any value directly but may save the evaluation results to out_file if provided.
    """

    result_files = result_dir.glob('**/*_result.csv')
    subject_to_file = {file_to_subject(f): f for f in result_files}
    if subjects is not None:
        subject_to_file = {s: f for s, f in subject_to_file.items() if s in subjects}

    print('--------------------------\nAccuracy by subject:')
    out_rows = []
    for subject in sorted(subject_to_file.keys()):
        result_df = pd.read_csv(subject_to_file[subject], sep=',', encoding='utf-8')
        acc = accuracy_score(y_true=result_df['label'], y_pred=result_df['prediction'])
        true_pos = sum(result_df['prediction'] == result_df['label'])
        num_labels = len(result_df['label'])
        out_rows.append({'subject': subject, 'true_pos': true_pos,
                         'num_labels': num_labels, 'accuracy': acc})
        print(f'{subject}: {acc:#.3}')

    sum_true_pos = sum(row['true_pos'] for row in out_rows)
    sum_labels = sum(row['num_labels'] for row in out_rows)
    micro_avg_acc = sum_true_pos / sum_labels

    print(f'---------------------\nMicro-averaged accuracy: {micro_avg_acc:#.3}')

    if out_file is not None:
        out_df = pd.DataFrame(out_rows)
        out_df.to_csv(out_file, sep=',', encoding='utf-8')