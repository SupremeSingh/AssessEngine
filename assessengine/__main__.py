"""Run with python -m assessengine --help."""
import argparse
import importlib.metadata
import json
import logging
import platform
import subprocess
import time
from pathlib import Path

from .core import (append_jsonl, digest, read_jsonl,
                   split_response, write_json, write_jsonl)

from .sas import DATASET, REVISION, SETS, prepare_records, response_prompt
MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
LOG = logging.getLogger('assessengine')


def revision(repo, repo_type='model'):
    from huggingface_hub import HfApi
    return HfApi().repo_info(repo, repo_type=repo_type).sha


def manifest(path, config):
    """Never silently mix incompatible runs when resuming."""
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['config'] != config:
            raise ValueError(f'Configuration changed for {path}; use a new directory.')
        return
    packages = {}
    for name in ('torch', 'transformers', 'datasets', 'bitsandbytes', 'huggingface-hub'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    write_json(path, {'config': config, 'python': platform.python_version(),
                      'packages': packages, 'git_commit': git.stdout.strip() or None})


def prepare(args):
    from datasets import load_dataset
    output = Path(args.data)
    output.mkdir(parents=True, exist_ok=True)
    info_path = output / 'dataset.json'
    saved = json.loads(info_path.read_text())['config'] if info_path.exists() else {}
    sha = args.revision or saved.get('revision') or REVISION
    manifest(info_path, {'dataset': DATASET, 'revision': sha, 'essay_sets': list(SETS),
                         'humans_per_question': args.humans_per_question,
                         'min_human_words': 50, 'seed': args.seed, 'preparation_version': 1})
    dataset = load_dataset(DATASET, revision=sha)
    rows = [dict(row, _split=name) for name, part in dataset.items() for row in part]
    records, coverage = prepare_records(rows, args.humans_per_question, args.seed)
    write_jsonl(output / 'questions.jsonl', records)
    write_json(output / 'coverage.json', coverage)
    LOG.info('Prepared %d SAS questions and %d eligible human answers', len(records),
             sum(len(q['humans']) for q in records))


def generate(args):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, set_seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Generation requires a Slurm allocation with an NVIDIA GPU.')
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    minimum = 20 if args.precision == '4bit' else 40
    if memory < minimum:
        raise RuntimeError(f'GPU has {memory:.1f} GiB; request a GPU with at least {minimum} GiB.')
    source = Path(args.data) / 'questions.jsonl'
    questions = read_jsonl(source)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    manifest_path = run / 'generation.json'
    saved = json.loads(manifest_path.read_text())['config'] if manifest_path.exists() else {}
    sha = args.revision or saved.get('revision') or revision(MODEL)
    config = {'model': MODEL, 'revision': sha, 'questions_sha256': digest(source),
              'samples': args.samples, 'seed': args.seed, 'precision': args.precision,
              'max_new_tokens': args.max_new_tokens, 'temperature': 0.6, 'top_p': 0.95,
              'prompt_mode': 'sas_question_context_and_length_v1', 'top_k': 0,
              'answer_min_words': args.answer_min_words, 'answer_max_words': args.answer_max_words}
    manifest(manifest_path, config)
    # Persist the exact human/question inputs alongside generated responses.
    write_jsonl(run / 'questions.jsonl', questions)
    destination = run / 'answers.jsonl'
    existing = read_jsonl(destination) if destination.exists() else []
    completed = {row['id'] for row in existing}
    work = [(q, sample) for q in questions for sample in range(args.samples)
            if f"{q['question_id']}:{sample}" not in completed]
    if not work:
        LOG.info('All responses already generated.')
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=sha, trust_remote_code=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    options = {'torch_dtype': dtype, 'device_map': {'': 0}, 'trust_remote_code': False}
    if args.precision == '4bit':
        options['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=sha, **options).eval()
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    for index, (question, sample) in enumerate(work):
        # A per-response seed makes interruption/resumption independent of RNG history.
        seed = args.seed + int(question['question_id'][:8], 16) + sample
        set_seed(seed % (2**32))
        messages = [{'role': 'user', 'content': response_prompt(
            question, args.answer_min_words, args.answer_max_words)}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors='pt').to('cuda:0')
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=True, temperature=0.6, top_p=0.95,
                                    max_new_tokens=args.max_new_tokens, top_k=0,
                                    pad_token_id=tokenizer.eos_token_id)
        tokens = output[0, inputs['input_ids'].shape[1]:].tolist()
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        parsed = split_response(raw, bool(tokens and tokens[-1] in eos_ids))
        row = {'id': f"{question['question_id']}:{sample}",
               'question_id': question['question_id'], 'sample': sample, 'seed': seed,
               'prompt': rendered, 'raw_response': raw, **parsed,
               'gpu': torch.cuda.get_device_name(0),
               'generated_tokens': len(tokens), 'seconds': round(time.monotonic() - started, 3)}
        append_jsonl(destination, row)
        LOG.info('%d/%d %s: %s (%d tokens, %.1fs)', index + 1, len(work),
                 row['id'], row['status'], len(tokens), row['seconds'])


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Prepare five ASAP-SAS science/biology prompts')
    prep.add_argument('--data', default='data/sas')
    prep.add_argument('--humans-per-question', type=int, default=50)
    prep.add_argument('--seed', type=int, default=42)
    prep.add_argument('--revision', help='Override the pinned Hugging Face dataset commit')
    gen = commands.add_parser('generate', help='Generate resumable DeepSeek answers on one GPU')
    gen.add_argument('--data', default='data/sas')
    gen.add_argument('--run', default='runs/sas-baseline')
    gen.add_argument('--samples', type=int, default=10)
    gen.add_argument('--seed', type=int, default=42)
    gen.add_argument('--max-new-tokens', type=int, default=4096)
    gen.add_argument('--answer-min-words', type=int, default=50)
    gen.add_argument('--answer-max-words', type=int, default=100)
    gen.add_argument('--precision', choices=['4bit', '16bit'], default='4bit')
    gen.add_argument('--revision', help='Hugging Face model commit')
    args = parser.parse_args()
    for field in ('samples', 'max_new_tokens', 'humans_per_question'):
        if hasattr(args, field) and getattr(args, field) < 1:
            parser.error(f'--{field.replace("_", "-")} must be positive')
    if args.command == 'generate' and not 50 <= args.answer_min_words <= args.answer_max_words:
        parser.error('Use 50 <= --answer-min-words <= --answer-max-words')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
