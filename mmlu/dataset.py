import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable

import pandas as pd

CHOICES = ['A', 'B', 'C', 'D']
DEFAULT_HEADER = 'The following are multiple choice questions (with answers) about'
DEFAULT_ANSWER = 'Answer'


@dataclass
class Dataset:
    subject: str
    prompt_header: str
    prompt_answer: str
    dev_df: pd.DataFrame
    test_df: pd.DataFrame

    def __len__(self):
        return len(self.test_df)

    @classmethod
    def from_dir(cls, data_dir: Path, subject: str) -> 'Dataset':
        dev_file = Path(data_dir / 'dev' / f'{subject}_dev.csv')
        test_file = Path(data_dir / 'test' / f'{subject}_test.csv')
        dev_df = pd.read_csv(dev_file, header=None, sep=',', encoding='utf-8', dtype=str, na_filter=False)
        test_df = pd.read_csv(test_file, header=None, sep=',', encoding='utf-8', dtype=str, na_filter=False)
        prompt_header = f'{DEFAULT_HEADER} {subject.replace("_", " ")}.\n\n'
        prompt_answer = DEFAULT_ANSWER
        if (data_dir / 'subjects.json').is_file():
            with open(data_dir / 'subjects.json', 'r', encoding='utf-8') as f:
                subjects = json.load(f)
            prompt_header = f'{subjects["header"]} {subjects["subjects"][subject]}.\n\n'
            prompt_answer = subjects['answer']
        return Dataset(test_df=test_df,
                       dev_df=dev_df,
                       subject=subject,
                       prompt_header=prompt_header,
                       prompt_answer=prompt_answer)


def file_to_subject(file: Path) -> str:
    extension = file.stem.split('_')[-1]
    return file.stem.replace(f'_{extension}', '')


def get_subjects(data_dir: Path) -> List[str]:
    files = list((data_dir / 'test').glob('**/*.csv'))
    subjects = [file_to_subject(f) for f in files]
    if len(subjects) == 0:
        raise ValueError(f'No test files found in: {data_dir}')
    return sorted(subjects)


def standard_token_counter(prompt: str) -> int:
    """
    Estimates the number of tokens for a prompt as number of characters) // 4
    """
    return len(prompt) // 4


def gen_prompt(dataset: Dataset,
               index: int,
               k_shot: int = 0,
               token_counter: Callable[[str], int] = None,
               max_tokens: Optional[int] = None):
    """
    Generates a prompt for a given index from the dataset.

    Args:
        dataset (Dataset): The Dataset object containing the data.
        index (int): The index of the example for which to generate the prompt.
        k_shot (int): The number of training examples (k-shot) to include in the prompt. Defaults to 0.
        token_counter (Callable[[str], int], optional): A callable function that takes a string as input and returns
                                                        the number of tokens in it. If None, the number of tokens
                                                        will be estimated by len(prompt) // 4. Defaults to None.
        max_tokens (int, optional): The maximum number of tokens allowed in the generated prompt.
                                    If the total tokens exceed this limit, the prompt will be truncated.
                                    Defaults to None.

    Returns:
        str: The generated prompt as a string.

    Note:
        This function constructs a prompt by concatenating the subject, a question, and additional k-shot
        training examples (if provided).
    """

    if token_counter is None:
        token_counter = standard_token_counter
    prompt = dataset.prompt_header
    question = _format_question(df=dataset.test_df,
                                index=index,
                                prompt_answer=dataset.prompt_answer,
                                include_answer=False)
    sum_tokens = token_counter(prompt) + token_counter(question) + 1

    # add training examples if enough tokens are left
    for k in range(k_shot):
        example = _format_question(df=dataset.dev_df,
                                   index=k,
                                   prompt_answer=dataset.prompt_answer,
                                   include_answer=True)
        sum_tokens += token_counter(example)
        if max_tokens is not None and sum_tokens >= max_tokens:
            break
        prompt += example

    prompt += question
    return prompt


def get_label(dataset: Dataset, index: int) -> str:
    return dataset.test_df.iloc[index, len(CHOICES) + 1]


def read_or_create_result_df(result_file: Path, dataset: Dataset) -> pd.DataFrame:
    try:
        result_df = pd.read_csv(result_file, sep=',', encoding='utf-8', dtype=str,
                                keep_default_na=False)
    except Exception as e:
        result_df = dataset.test_df.copy(deep=True)
        result_df.columns = ['prompt', 'A', 'B', 'C', 'D', 'label']
        result_df['prediction'] = ''
        result_df.to_csv(result_file, sep=',', encoding='utf-8', index=False)
    return result_df


def _format_question(df: pd.DataFrame,
                     index: int,
                     prompt_answer: str,
                     include_answer=True):
    prompt = df.iloc[index, 0]
    for j, choice in enumerate(CHOICES):
        prompt += f'\n{choice}. {df.iloc[index, j+1]}'
    prompt += f'\n{prompt_answer}:'
    if include_answer:
        prompt += f' {df.iloc[index, len(CHOICES)+1]}\n\n'
    return prompt