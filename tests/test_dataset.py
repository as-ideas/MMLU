import unittest
import os
import unittest
from pathlib import Path

from mmlu.dataset import Dataset, gen_prompt, get_label


def read_txt(file: Path) -> str:
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()


class TestDataset(unittest.TestCase):

    def setUp(self) -> None:
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.resources_path = Path(test_path) / 'resources'
        self.data_path = Path(test_path) / 'resources' / 'test_data'
        self.test_file = self.data_path / 'test' / 'astronomy_test.csv'
        self.dev_file = self.data_path / 'dev' / 'astronomy_dev.csv'

    def test_create_from_dir(self):
        dataset = Dataset.from_dir(self.data_path, 'astronomy')
        self.assertEqual(5, len(dataset.test_df))
        self.assertEqual(5, len(dataset.dev_df))
        self.assertEqual('Why is the sky blue?', dataset.test_df.iloc[2, 0])
        self.assertEqual('Why is Mars red?', dataset.dev_df.iloc[4, 0])

    def test_gen_prompt(self):
        dataset = Dataset(test_file=self.test_file,
                          dev_file=self.dev_file,
                          subject='astronomy')

        prompt_k0 = gen_prompt(dataset, index=0, k_shot=0)
        expected_k0 = read_txt(self.resources_path / 'astronomy_prompt_k0_expected.txt')
        self.assertEqual(expected_k0, prompt_k0)

        prompt_k1 = gen_prompt(dataset, index=0, k_shot=1)
        expected_k1 = read_txt(self.resources_path / 'astronomy_prompt_k1_expected.txt')
        self.assertEqual(expected_k1, prompt_k1)

    def test_get_label(self):
        dataset = Dataset(test_file=self.test_file,
                          dev_file=self.dev_file,
                          subject='astronomy')
        label = get_label(dataset, 0)
        self.assertEqual('A', label)


