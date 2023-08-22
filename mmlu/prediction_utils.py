from transformers import LogitsProcessor
from typing import Dict


class LogitBiasProcessor(LogitsProcessor):

    def __init__(self, logit_bias: Dict[int, float]):
        super(LogitBiasProcessor, self).__init__()
        self._logit_bias = logit_bias
        pass

    def __call__(self, input_ids, logits):
        for logit_id, bias in self._logit_bias.items():
            logits[:, logit_id] += bias
        return logits
