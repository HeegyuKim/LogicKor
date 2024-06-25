import os
import argparse
import pandas as pd
from transformers import pipeline
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_device', help=' : CUDA_VISIBLE_DEVICES', default='0')
parser.add_argument('--model', help=' : Model to evaluate', default='yanolja/EEVE-Korean-Instruct-2.8B-v1.0')
parser.add_argument('--revision', help=' : Model Revision to evaluate', default=None)
parser.add_argument('--model_len', help=' : Maximum Model Length', default=4096, type=int)
parser.add_argument('--eos_token', help=' : End of Sentence Token', default=None, type=str)
parser.add_argument('--output_dir', help=' : Output Directory', default='./output', type=str)
# parser.add_argument('--trust_remote_code/--not_trust_remote_code', help=' : Trust remote code?', default=False, type=bool)
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

generator = pipeline('text-generation', model=args.model, revision=args.revision, device_map="cuda:0", trust_remote_code=True)

if args.eos_token:
    generator.tokenizer.eos_token = args.eos_token

sampling_params = dict(
    temperature=0,
    top_p=1,
    top_k=-1,
    early_stopping=True,
    max_length=args.model_len,
    max_new_tokens=1024,
)

df_questions = pd.read_json('questions.jsonl', lines=True)

def format_single_turn_question(question):
    return [{'role': 'user', 'content': question[0]}]

single_turn_questions = df_questions['questions'].map(format_single_turn_question)
single_turn_outputs = []
for q in tqdm(single_turn_questions, desc='Single Turn Generation'):
    output = generator(q, **sampling_params)[0]['generated_text'][-1]['content'].strip()
    print(q)
    print('->')
    print(output)
    print()
    single_turn_outputs.append(output)

def format_double_turn_question(question, single_turn_output):
    return [
        {'role': 'user', 'content': question[0]},
        {'role': 'assistant', 'content': single_turn_output},
        {'role': 'user', 'content': question[1]}
        ]

multi_turn_questions = df_questions[['questions', 'id']].apply(
    lambda x: format_double_turn_question(x['questions'], single_turn_outputs[x['id'] - 1]),
    axis=1
) # bad code ig?

multi_turn_outputs = []
for q in tqdm(multi_turn_questions, desc='Multi Turn Generation'):
    output = generator(q, **sampling_params)[0]['generated_text'][-1]['content'].strip()
    print(q)
    print('->')
    print(output)
    print()
    multi_turn_outputs.append(output)

df_output = pd.DataFrame({
    'id': df_questions['id'],
    'category': df_questions['category'],
    'questions': df_questions['questions'],
    'outputs': list(zip(single_turn_outputs, multi_turn_outputs)),
    'references': df_questions['references']
})
df_output.to_json(
    os.path.join(args.output_dir, f'{str(args.model).replace("/", "_")}@{args.revision}.jsonl'),
    orient='records',
    lines=True,
    force_ascii=False
)
