from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

CHOICES = ['A', 'B', 'C', 'D']
PROMPT = 'The following are multiple choice questions (with answers) about'


@dataclass
class Dataset:
    subject: str
    dev_df: pd.DataFrame
    test_df: pd.DataFrame

    def __init__(self,
                 test_file: Path,
                 dev_file: Path,
                 subject: str) -> None:
        self.subject = subject
        self.test_df = pd.read_csv(test_file, header=None, sep=',', encoding='utf-8')
        self.dev_df = pd.read_csv(dev_file, header=None, sep=',', encoding='utf-8')

    def __len__(self):
        return len(self.test_df)

    @classmethod
    def from_dir(cls, data_dir: Path, subject: str) -> 'Dataset':
        test_file = Path(data_dir / 'test' / f'{subject}_test.csv')
        dev_file = Path(data_dir / 'dev' / f'{subject}_dev.csv')
        return Dataset(test_file=test_file, dev_file=dev_file, subject=subject)


def get_subjects(data_dir: Path) -> List[str]:
    files = (data_dir / 'test').glob('**/*.csv')
    subjects = [f.stem.replace('_test', '') for f in files]
    return sorted(subjects)


def gen_prompt(dataset: Dataset,
               index: int,
               k_shot: int = 0):
    prompt = f'{PROMPT} {dataset.subject}.\n\n'
    for k in range(k_shot):
        example = _format_question(dataset.dev_df, index, include_answer=True)
        prompt += example
    question = _format_question(dataset.test_df, index, include_answer=False)
    prompt += question
    return prompt


def get_label(dataset: Dataset, index: int) -> str:
    return dataset.test_df.iloc[index, len(CHOICES) + 1]


def read_or_create_result_df(result_file: Path, dataset: Dataset) -> pd.DataFrame:
    try:
        result_df = pd.read_csv(result_file, sep=',', encoding='utf-8')
    except Exception as e:
        labels = [get_label(dataset, i) for i in range(len(dataset))]
        rows = [{'label': label, 'prediction': None} for label in labels]
        result_df = pd.DataFrame(rows)
        result_df.to_csv(result_file, sep=',', encoding='utf-8')
    return result_df



def _format_question(df: pd.DataFrame,
                     index: int,
                     include_answer=True):
    prompt = df.iloc[index, 0]
    for j, choice in enumerate(CHOICES):
        prompt += f'\n{choice}. {df.iloc[index, j+1]}'
    prompt += '\nAnswer:'
    if include_answer:
        prompt += f' {df.iloc[index, len(CHOICES)+1]}\n\n'
    return prompt