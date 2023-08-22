import unittest
import torch

from mmlu.prediction_utils import LogitBiasProcessor


class TestLogitBiasProcessor(unittest.TestCase):

    def test_process_happy_path(self):
        logit_processor = LogitBiasProcessor(logit_bias={0: 0.5, 2: -0.3})
        input_ids = torch.tensor([[1, 2, 3]])
        logits = torch.tensor([[0.2, 0.4, 0.6]])
        processed_logits = logit_processor(input_ids, logits)
        expected_logits = torch.tensor([[0.7, 0.4, 0.3]])  # Applying bias to logit 0 and 2
        self.assertTrue(torch.allclose(processed_logits, expected_logits, atol=1e-10))