import argparse

import json
import os
from pathlib import Path

import pandas as pd
import requests
import uuid
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

from mmlu.dataset import Dataset, get_subjects, DEFAULT_HEADER, DEFAULT_ANSWER

AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT', default='https://api.cognitive.microsofttranslator.com/translate')
AZURE_REGION = os.getenv('AZURE_REGTION', default='germanywestcentral')
AZURE_KEY = os.getenv('AZURE_KEY')


class AzureTranslator:

    def __init__(self,
                 url: str,
                 source_lang: str,
                 target_lang: str,
                 retries: int = 3,
                 timeout: int = 30):
        self.url = url
        self.retries = retries
        self.timeout = timeout
        self.headers = {'Ocp-Apim-Subscription-Key': AZURE_KEY,
                        'Ocp-Apim-Subscription-Region': AZURE_REGION,
                        'Content-type': 'application/json',
                        'X-ClientTraceId': str(uuid.uuid4())}
        self.params = {'api-version': '3.0',
                       'from': source_lang,
                       'to': target_lang}
        self.retry_strategy = Retry(total=self.retries,
                                    backoff_factor=0.3,
                                    status_forcelist=[429, 500, 502, 503, 504])

    def __call__(self, text: str) -> str:
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=self.retry_strategy)
        session.mount("https://", adapter)
        body = [{'text': text}]
        response = session.post(self.url,
                                params=self.params,
                                headers=self.headers,
                                json=body,
                                timeout=self.timeout)
        response.raise_for_status()
        return response.json()[0]['translations'][0]['text']


def run_translation(df: pd.DataFrame, target_file: Path) -> None:
    if target_file.is_file():
        return
    df.columns = ['prompt', 'A', 'B', 'C', 'D', 'label']
    rows_trans = []
    for j, row in tqdm(df.iterrows(), total=len(df)):
        test_row_trans = {'prompt': translator(row['prompt']),
                          'A': translator(row['A']),
                          'B': translator(row['B']),
                          'C': translator(row['C']),
                          'D': translator(row['D']),
                          'label': row['label']}
        rows_trans.append(test_row_trans)
    df_trans = pd.DataFrame(rows_trans, columns=['prompt', 'A', 'B', 'C', 'D', 'label'])
    df_trans.to_csv(target_file, index=False,
                    header=False, sep=',', encoding='utf-8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--target_dir', type=str, help='Target directory to store translated data.')
    parser.add_argument('--lang', type=str, help='Target language for translation.')
    args = parser.parse_args()
    print(args)
    translator = AzureTranslator(url=AZURE_ENDPOINT,
                                 source_lang='en',
                                 target_lang=args.lang,
                                 retries=3,
                                 timeout=30)

    target_dir = Path(args.target_dir)
    data_dir = Path(args.data_dir)
    (target_dir / 'dev').mkdir(parents=True, exist_ok=True)
    (target_dir / 'test').mkdir(parents=True, exist_ok=True)
    subjects = get_subjects(data_dir=data_dir)

    print(f'Translate header and subject names, will dump the result to {target_dir / "subjects.json"}')
    subjects_translated = {'header': translator(DEFAULT_HEADER),
                           'answer': translator(DEFAULT_ANSWER),
                           'subjects': {}}
    for subject in tqdm(subjects, total=len(subjects)):
        subject_trans = translator(subject.replace('_', ' '))
        subjects_translated['subjects'][subject] = subject_trans
    with open(target_dir / 'subjects.json', 'w', encoding='utf-8') as f:
        json.dump(subjects_translated, f)

    for i, subject in enumerate(subjects):
        print(f'Translate subject {subject} ({i}/{len(subjects)})')
        dataset = Dataset.from_dir(data_dir=data_dir, subject=subject)
        target_dir.mkdir(parents=True, exist_ok=True)
        run_translation(dataset.dev_df, target_dir / 'dev' / f'{subject}_dev.csv')
        run_translation(dataset.test_df, target_dir / 'test' / f'{subject}_test.csv')
