import unittest
import os
import unittest
from pathlib import Path

from mmlu.dataset import Dataset, gen_prompt


def read_txt(file: Path) -> str:
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()


class TestDataset(unittest.TestCase):

    def setUp(self) -> None:
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.resources_path = Path(test_path) / 'resources'

    def test_gen_prompt(self):
        test_file = self.resources_path / 'astronomy_test_single.csv'
        dev_file = self.resources_path / 'astronomy_dev_single.csv'
        dataset = Dataset(test_file=test_file, dev_file=dev_file, subject='astronomy')

        prompt_k0 = gen_prompt(dataset, index=0, k_shot=0)
        expected_k0 = read_txt(self.resources_path / 'astronomy_prompt_k0_expected.txt')
        self.assertEqual(expected_k0, prompt_k0)

        prompt_k1 = gen_prompt(dataset, index=0, k_shot=1)
        expected_k1 = read_txt(self.resources_path / 'astronomy_prompt_k1_expected.txt')
        self.assertEqual(expected_k1, prompt_k1)
