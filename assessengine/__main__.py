"""Run with python -m assessengine --help."""
import argparse
import csv
import importlib.metadata
import json
import logging
import platform
import subprocess
import time
from pathlib import Path

from .core import (append_jsonl, digest, question_records, read_jsonl,
                   split_response, write_json, write_jsonl)

DATASET = 'nkazi/MohlerASAG'
MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
DETECTOR = 'desklib/ai-text-detector-v1.01'
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
    sha = args.revision or saved.get('revision') or revision(DATASET, 'dataset')
    manifest(info_path, {'dataset': DATASET, 'revision': sha, 'subset': 'raw',
                         'split': 'open_ended', 'questions': args.questions, 'seed': args.seed})
    ds = load_dataset(DATASET, name='raw', split='open_ended', revision=sha)
    records = question_records(list(ds), args.questions, args.seed)
    write_jsonl(output / 'questions.jsonl', records)
    LOG.info('Prepared %d questions and %d human answers', len(records),
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
              'prompt_mode': 'exact_question_only', 'top_k': 0}
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
        messages = [{'role': 'user', 'content': question['question']}]
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


def score(args):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

    class Detector(PreTrainedModel):
        """Desklib checkpoint layout: DeBERTa, masked mean pooling, one logit."""
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = torch.nn.Linear(config.hidden_size, 1)
            self.post_init()

        def forward(self, input_ids, attention_mask):
            hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.classifier(pooled)

    run = Path(args.run)
    source = run / 'questions.jsonl'
    answers_path = run / 'answers.jsonl'
    info_path = run / 'detector.json'
    saved = json.loads(info_path.read_text())['config'] if info_path.exists() else {}
    sha = args.revision or saved.get('revision') or revision(DETECTOR)
    manifest(info_path, {'detector': DETECTOR, 'revision': sha,
                         'questions_sha256': digest(source),
                         'answers_sha256': digest(answers_path), 'max_tokens': 768})
    rows = []
    for question in read_jsonl(source):
        for human in question['humans']:
            rows.append({'id': human['id'], 'question_id': question['question_id'],
                         'authorship': 'human', 'text': human['student_answer'],
                         'grade': human['score_avg']})
    for answer in read_jsonl(answers_path):
        if answer['status'] == 'complete':
            rows.append({'id': answer['id'], 'question_id': answer['question_id'],
                         'authorship': 'ai', 'text': answer['answer'], 'grade': None})
    tokenizer = AutoTokenizer.from_pretrained(DETECTOR, revision=sha, trust_remote_code=False)
    model, loading = Detector.from_pretrained(
        DETECTOR, revision=sha, trust_remote_code=False, output_loading_info=True)
    if loading['missing_keys'] or loading['unexpected_keys'] or loading['mismatched_keys']:
        raise RuntimeError(f'Detector checkpoint layout mismatch: {loading}')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    for row in rows:
        row['words'] = len(row['text'].split())
        row['tokens'] = len(tokenizer.encode(row['text'], add_special_tokens=True))
        row['detector_truncated'] = row['tokens'] > 768
        # Empty responses cannot provide authorship evidence.
        row['ai_score'] = None
        if row['text'].strip():
            inputs = tokenizer(row['text'], truncation=True, max_length=768,
                               return_tensors='pt').to(device)
            with torch.inference_mode():
                row['ai_score'] = torch.sigmoid(model(
                    inputs['input_ids'], inputs['attention_mask'])).item()
    write_jsonl(run / 'scores.jsonl', rows)
    with (run / 'scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['id'])
        writer.writeheader()
        writer.writerows(rows)
    LOG.info('Scored %d answers', len(rows))


def report(args):
    from sklearn.metrics import roc_auc_score
    from statistics import mean, median
    run = Path(args.run)
    rows = read_jsonl(run / 'scores.jsonl')
    valid = [r for r in rows if r['ai_score'] is not None]
    humans = [r for r in valid if r['authorship'] == 'human']
    ai = [r for r in valid if r['authorship'] == 'ai']
    answers = read_jsonl(run / 'answers.jsonl')
    summary = {'human_answers': len(humans), 'ai_answers': len(ai),
               'excluded_generation_statuses': {},
               'threshold': 0.5, 'warning': 'Exploratory detector baseline, not genuine divergence. '
               'Short text, answer correctness, missing question context, length differences, '
               'and potential public-dataset contamination are uncontrolled.'}
    for answer in answers:
        status = answer['status']
        if status != 'complete':
            summary['excluded_generation_statuses'][status] = summary['excluded_generation_statuses'].get(status, 0) + 1
    summary['auroc'] = roc_auc_score(
        [int(r['authorship'] == 'ai') for r in valid],
        [r['ai_score'] for r in valid]) if humans and ai else None
    for label, group in [('human', humans), ('ai', ai)]:
        summary[label] = {'mean_ai_score': mean(r['ai_score'] for r in group) if group else None,
                          'fraction_flagged_ai': mean(r['ai_score'] >= 0.5 for r in group) if group else None,
                          'median_words': median(r['words'] for r in group) if group else None,
                          'under_50_words': sum(r['words'] < 50 for r in group),
                          'detector_truncated': sum(r['detector_truncated'] for r in group)}
    write_json(run / 'report.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Download raw Mohler responses; sample questions')
    prep.add_argument('--data', default='data/mohler')
    prep.add_argument('--questions', type=int, default=10)
    prep.add_argument('--seed', type=int, default=42)
    prep.add_argument('--revision', help='Hugging Face dataset commit; otherwise resolve and record')
    gen = commands.add_parser('generate', help='Generate resumable DeepSeek answers on one GPU')
    gen.add_argument('--data', default='data/mohler')
    gen.add_argument('--run', default='runs/baseline')
    gen.add_argument('--samples', type=int, default=3)
    gen.add_argument('--seed', type=int, default=42)
    gen.add_argument('--max-new-tokens', type=int, default=2048)
    gen.add_argument('--precision', choices=['4bit', '16bit'], default='4bit')
    gen.add_argument('--revision', help='Hugging Face model commit')
    detector = commands.add_parser('score', help='Score human answers and completed AI final answers')
    detector.add_argument('--run', default='runs/baseline')
    detector.add_argument('--revision', help='Hugging Face detector commit')
    rep = commands.add_parser('report', help='Write baseline detector metrics')
    rep.add_argument('--run', default='runs/baseline')
    args = parser.parse_args()
    for field in ('samples', 'max_new_tokens', 'questions'):
        if hasattr(args, field) and getattr(args, field) < 1:
            parser.error(f'--{field.replace("_", "-")} must be positive')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
