import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from mmlu.evaluation import predict_dataset


def read_txt(file: Path) -> str:
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()


def mock_predict_func(x: str) -> str:
    return 'A'


class TestEvaluate(unittest.TestCase):

    def setUp(self) -> None:
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.resources_path = Path(test_path) / 'resources'
        self.data_path = Path(test_path) / 'resources' / 'en_test_data'
        self.temp_dir = TemporaryDirectory(prefix='TestEvaluateTmpDir')

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_predict_dataset_sequentially(self):

        result_dir = Path(self.temp_dir.name) / 'sequential_results'

        # predict without concurrency
        predict_dataset(data_dir=self.data_path,
                        result_dir=result_dir,
                        subjects=['astronomy_for_testing'],
                        predict_function=mock_predict_func,
                        k_shot=0,
                        n_workers=0)

        results_df = pd.read_csv(result_dir / 'astronomy_for_testing_result.csv')

        labels = results_df['label'].tolist()
        self.assertEqual(['A', 'D', 'C', 'C', 'D'], labels)

        predictions = results_df['prediction'].tolist()
        print(['A', 'A', 'A', 'A', 'A', predictions])

    def test_predict_dataset_multithreaded(self):

        result_dir = Path(self.temp_dir.name) / 'multithreaded_results'

        # predict without concurrency
        predict_dataset(data_dir=self.data_path,
                        result_dir=result_dir,
                        subjects=['astronomy_for_testing'],
                        predict_function=mock_predict_func,
                        k_shot=0,
                        n_workers=2)

        results_df = pd.read_csv(result_dir / 'astronomy_for_testing_result.csv')

        labels = results_df['label'].tolist()
        self.assertEqual(['A', 'D', 'C', 'C', 'D'], labels)

        predictions = results_df['prediction'].tolist()
        print(['A', 'A', 'A', 'A', 'A', predictions])





