import threading
from typing import Tuple, Callable


class PredictionWithTimeout:

    def __init__(self,
                 func: Callable[[str], str],
                 timeout_s: float = 50,
                 retries: int = 3) -> str:
        self._func = func
        self._timeout_s = timeout_s
        self._retries = retries

    def __call__(self, prompt: str) -> str:
        func_result = None
        result = None

        def run_func():
            nonlocal func_result
            func_result = self._func(prompt)

        for retry in range(self._retries):
            #print(id(self), 'retry: ', retry)
            thread = threading.Thread(target=run_func)
            thread.start()
            thread.join(self._timeout_s)
            if func_result is not None:
                result = func_result
                break

        if result is None:
            raise RuntimeError(f'Max retries ({self._retries}) reached!')

        return result


class PredictionWorker:

    def __init__(self, predictor: Callable[[str], str]) -> None:
        self.predictor = predictor

    def __call__(self, prompt_index: Tuple[str, int]) -> Tuple[str, int]:
        prompt, index = prompt_index
        pred = self.predictor(prompt)
        return pred, index