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
        self.data_path = Path(test_path) / 'resources' / 'en_test_data'
        self.test_file = self.data_path / 'test' / 'astronomy_for_testing_test.csv'
        self.dev_file = self.data_path / 'dev' / 'astronomy_for_testing_dev.csv'

    def test_create_from_dir(self):
        dataset = Dataset.from_dir(self.data_path, 'astronomy_for_testing')
        self.assertEqual(5, len(dataset.test_df))
        self.assertEqual(5, len(dataset.dev_df))
        self.assertEqual('Why is the sky blue?', dataset.test_df.iloc[2, 0])
        self.assertEqual('Why is Mars red?', dataset.dev_df.iloc[4, 0])

    def test_gen_prompt(self):
        dataset = Dataset.from_dir(data_dir=self.data_path, subject='astronomy_for_testing')

        prompt_k0 = gen_prompt(dataset, index=0, k_shot=0)
        expected_k0 = read_txt(self.resources_path / 'astronomy_prompt_k0_expected.txt')
        self.assertEqual(expected_k0, prompt_k0)

        prompt_k1 = gen_prompt(dataset, index=0, k_shot=1)
        expected_k1 = read_txt(self.resources_path / 'astronomy_prompt_k1_expected.txt')
        self.assertEqual(expected_k1, prompt_k1)

        prompt_k5 = gen_prompt(dataset, index=0, k_shot=5)
        expected_k5 = read_txt(self.resources_path / 'astronomy_prompt_k5_expected.txt')
        self.assertEqual(expected_k5, prompt_k5)

        # Below we set a token limit for the prompt of 1300 characters, so some of the k-shot examples should
        # be excluded for the prompt to fit the token limit
        prompt_k5_max100 = gen_prompt(dataset, index=0, k_shot=5, max_tokens=1300,
                                      token_counter=lambda x: len(x))
        expected_k5_max100 = read_txt(self.resources_path / 'astronomy_prompt_k5_maxtokens1300_expected.txt')
        self.assertEqual(expected_k5_max100, prompt_k5_max100)

    def test_get_label(self):
        dataset = Dataset.from_dir(data_dir=self.data_path, subject='astronomy_for_testing')
        label = get_label(dataset, 0)
        self.assertEqual('A', label)


