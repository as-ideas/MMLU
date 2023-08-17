# MMLU Evaluation

This repository contains code for running the [MMLU](https://arxiv.org/abs/2009.03300) (Massive Multitask Language Understanding) evaluation of large language models.
It is recoded from scratch following the logic of the [original repo](https://github.com/hendrycks/test) with following imrovements:

- **Accellerated inference**: Using multithreaded API calls.
- **Enhanced stability**: Added timeouts and retries for API calls.
- **Modularity**: You can easily evaluate your custom LLM (see [Evaluate your custom model](#evaluate-custom)).

## Setup
1. Download the dataset [here](https://people.eecs.berkeley.edu/~hendrycks/data.tar)

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure you have the necessary environment variables if you want to use OpenAI or Azure models. E.g. for Azure:
```bash
export OPENAI_API_BASE=https://your-azure-endpoint.com 
export OPENAI_API_KEY=your-azure-key
```

## Usage

Run the evaluation code. The results are stored as *.csv files in the given directory.

```bash
python evaluate_azure.py --data_dir path-to-data --result_dir path-to-results --k_shot 0
```

## Evaluate your custom model <a id="evaluate-custom"></a>

To evaluate a custom LLE simply use the following template and replace the predict_function by your own Callable:

```python
from pathlib import Path
from mmlu.evaluation import predict_dataset, evaluate_results


def predict_function(prompt: str) -> str:
    return 'A'


if __name__ == '__main__':
    data_dir = Path('data')
    result_dir = Path('results')
    predict_dataset(data_dir=data_dir,
                    result_dir=result_dir,
                    predict_function=predict_function,
                    k_shot=0)
    evaluate_results(result_dir=result_dir)
```

## Languages other than English

### Evaluating on other languages


We will provide additional datasets (starting with German) that are translated via Azure and can be used ad-hoc with the standard evaluation script - simply point to the translated data.

A translated dataset is formatted in the same way as the original dataset but contains an additional file ```subjects.json``` that includes the translated prompt header and subjects:
```
data_de/
├── dev/
├── test/
├── subjects.json
```

For German, the ```subjects.json``` looks like:

```json
{
  "header": "Im Folgenden finden Sie Multiple-Choice-Fragen (mit Antworten) zum Thema",
  "answer": "Antwort", 
  "subjects": {
    "abstract_algebra": "abstrakte Algebra", 
    "astronomy": "Astronomie",
    ...
  }
}
```

### Translating the dataset to another language

You can use the translation script that calls the Azure translation service:

```bash
export AZURE_ENDPOINT=your-azure-translation-endpoint
export AZURE_KEY=your-azure-key
export AZURE_REGION=your-azure-region
PYTHONPATH=. python mmlu/translate --data_dir data --target_dir /tmp/data_de --lang de
```

The translated data will be stored in ```target_dir``` in the format described above. Note that only ```dev``` and ```test``` data will be translated.


## References

* [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
* [Original Implementation](https://github.com/hendrycks/test)

