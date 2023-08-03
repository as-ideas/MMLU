import os
from pathlib import Path

import openai

from mmlu.evaluation import predict_dataset, evaluate_results

openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview'
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv('OPENAI_API_KEY')


def predict_azure_gpt_35_turbo(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        engine='gpt-35-turbo',
        messages=[
            {'role': 'system', 'content': ''},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        # Logit biases for tokens [' A', ' B', ' C', ' D', 'A', 'B', 'C', 'D']
        logit_bias={362: 100, 426: 100, 356: 100, 423: 100,
                    32: 100, 33: 100, 34: 100, 35: 100},
        stop=None)
    pred = response['choices'][0]['message']['content'].replace(' ', '')
    return pred


if __name__ == '__main__':
    data_dir = Path('/Users/cschaefe/datasets/nlp/mmlu/data')
    result_dir = Path(f'results/azure_gpt_35_turbo_logit_bias2')

    predict_dataset(data_dir=data_dir,
                    result_dir=result_dir,
                    predict_function=predict_azure_gpt_35_turbo,
                    subjects=['anatomy'],
                    k_shot=0,
                    n_workers=2,
                    timeout_s=50,
                    retries=3)

    evaluate_results(result_dir=result_dir, out_file=Path('/tmp/chatgpt.csv'))
