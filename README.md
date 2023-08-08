# MMLU Repository

This repository contains code for the **MMLU** (Machine Learning Model Utilization) project, which focuses on evaluating of LLMs. 

## Setup
Download the dataset [here](https://people.eecs.berkeley.edu/~hendrycks/data.tar)

Install the required dependencies:

```bash
pip install openai argparse
```

Ensure you have the necessary environment variables if you want to use OpenAI or Azure models. E.g. for Azure:
```bash
export OPENAI_API_BASE=https://your-azure-endpoint.com 
export OPENAI_API_KEY=your-azure-key
```

## Usage

Run the evaluation code. The results are stored as *.csv files in the given directory.

```bash
python evaluate_azure.py --data_dir path-to-data --result_dir path-to-results --k_shot 0
```

## Evaluating your custom model

You can easily implement evaluation of any custom language model. Simply use the following template and replace the predict_function by your own callable:

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


