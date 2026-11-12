"""Export saved answers and run the official SnapJudge batch client."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .core import digest, read_jsonl, write_json, write_jsonl
from .selection import select_answers
from .detector_scores import continuous_scores

CLIENT_URL = 'https://nlplab.cs.duke.edu/snapjudge/batch/snapjudge_batch.py'
MIN_WORDS = 50


def export(run, allow_partial=False):
    run = Path(run)
    generation_manifest = run / 'generation.json'
    if generation_manifest.exists() and not allow_partial:
        config = json.loads(generation_manifest.read_text())['config']
        if config.get('prompt_mode') == 'sas_question_context_and_length_v2':
            coverage = run / 'generation_coverage.json'
            if not coverage.exists() or not json.loads(coverage.read_text())['all_quotas_met']:
                raise ValueError('AI quotas are not met; finish generation before exporting or scoring.')
    output = run / 'snapjudge'
    inputs = output / 'documents'
    questions, answers = run / 'questions.jsonl', run / 'answers.jsonl'
    sources = {'questions_sha256': digest(questions), 'answers_sha256': digest(answers),
               'min_words': MIN_WORDS, 'format_version': 1}
    if allow_partial:
        sources.update(format_version=2, selection_policy='balanced_saved_answers_v1')
    manifest = output / 'export.json'
    if manifest.exists() and json.loads(manifest.read_text()) != sources:
        raise ValueError('Inputs changed since export; use a new run directory to avoid stale results.')
    records = []
    for question in read_jsonl(questions):
        for human in question['humans']:
            records.append({'id': human['id'], 'question_id': question['question_id'],
                            'authorship': 'human', 'text': human['student_answer'],
                            'generation_status': None})
    for answer in read_jsonl(answers):
        records.append({'id': answer['id'], 'question_id': answer['question_id'],
                        'authorship': 'ai', 'text': answer['answer'],
                        'generation_status': answer['status']})
    if allow_partial:
        records = [{**row, 'generation_status': 'complete' if row['authorship'] == 'ai' else None}
                   for row in select_answers(run, allow_partial=True)]
    inputs.mkdir(parents=True, exist_ok=True)
    for row in records:
        row['words'] = len(row['text'].split())
        reason = None
        if row['authorship'] == 'ai' and row['generation_status'] != 'complete':
            reason = 'incomplete_generation'
        elif row['words'] < MIN_WORDS:
            reason = 'under_50_words'
        row['excluded_reason'], row['file'] = reason, None
        if reason is None:
            name = hashlib.sha256(f"{row['authorship']}:{row['id']}".encode()).hexdigest() + '.txt'
            row['file'] = name
            path = inputs / name
            if path.exists() and path.read_text(encoding='utf-8') != row['text']:
                raise ValueError(f'Exported document changed: {path}')
            path.write_text(row['text'], encoding='utf-8')
    expected = {r['file'] for r in records if r['file']}
    if {p.name for p in inputs.iterdir()} != expected:
        raise ValueError('Unexpected files in SnapJudge documents folder; use a clean run.')
    write_json(manifest, sources)
    write_jsonl(output / 'index.jsonl', records)
    counts = Counter(f"{r['authorship']}:{r['excluded_reason'] or 'eligible'}" for r in records)
    write_json(output / 'coverage.json', dict(counts))
    print(json.dumps(dict(counts), indent=2))
    return output


def report(run):
    output = Path(run) / 'snapjudge'
    index = read_jsonl(output / 'index.jsonl')
    # The official batch client writes UTF-8 with a BOM for spreadsheet compatibility.
    # utf-8-sig accepts files both with and without that marker.
    with (output / 'results.csv').open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'file', 'verdict', 'ai_generated_share', 'error'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Unexpected SnapJudge CSV columns: {reader.fieldnames}')
        results = list(reader)
    by_file = {}
    expected = {row['file'] for row in index if row['file']}
    for result in results:
        name = result['file']
        if name not in expected or name in by_file:
            raise ValueError(f'Unexpected or duplicate result: {name}')
        by_file[name] = result
    joined = [{**row, 'snapjudge': by_file.get(row['file'])} for row in index]
    for row in joined:
        row['continuous_scores'] = None
        if row['file']:
            raw_path = output/'results_json'/(row['file']+'.json')
            if raw_path.exists():
                raw = json.loads(raw_path.read_text())
                result = (raw.get('answer',{}).get('results') or [{}])[0]
                row['continuous_scores'] = continuous_scores(result)
    write_jsonl(output / 'scores.jsonl', joined)
    with (output/'continuous_scores.csv').open('w',newline='',encoding='utf-8') as stream:
        fields = ['id','question_id','authorship','words','mean_generated_decision_margin',
                  'mean_nonhuman_score','mean_generated_vs_polished_score','thr_nonhuman','thr_generated']
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in joined:
            values = {**row, **(row['continuous_scores'] or {})}
            writer.writerow({k:values.get(k) for k in fields})
    summary = {'detector': 'SnapJudge', 'min_words': MIN_WORDS,
               'score_definition': 'Fraction of windows classified AI-generated, not authorship probability.',
               'warning': 'Eligible answers only. Length filtering changes the sample; '
                          'short human answers are excluded. Calibrated on research papers; '
                          'this is not evidence of true human-LLM divergence.'}
    for label in ('human', 'ai'):
        group = [row for row in joined if row['authorship'] == label]
        successful = []
        for row in group:
            result = row['snapjudge']
            if result and not result.get('error'):
                share = float(result['ai_generated_share'])
                if not 0 <= share <= 1:
                    raise ValueError('Invalid SnapJudge generated-window share')
                successful.append((result, share))
        summary[label] = {
            'total': len(group), 'eligible': sum(not r['excluded_reason'] for r in group),
            'excluded': dict(Counter(r['excluded_reason'] for r in group if r['excluded_reason'])),
            'scored': len(successful),
            'errors': sum(bool(r['snapjudge'] and r['snapjudge'].get('error')) for r in group),
            'missing_results': sum(not r['excluded_reason'] and r['snapjudge'] is None for r in group),
            'verdicts': dict(Counter(r['verdict'] for r, _ in successful)),
            'mean_generated_window_share': (sum(s for _, s in successful) / len(successful)
                                             if successful else None)}
        margins = [r['continuous_scores']['mean_generated_decision_margin'] for r in group if r['continuous_scores']]
        summary[label]['continuous_scored'] = len(margins)
        summary[label]['mean_generated_decision_margin'] = sum(margins)/len(margins) if margins else None
    summary['continuous_score_note'] = ('Derived raw-logit decision margins are confidence proxies, not calibrated '
        'probabilities. Compare using the same detector/thresholds. Mean window margins for multi-window texts '
        'do not equal the document verdict rule.')
    write_json(output / 'report.json', summary)
    print(json.dumps(summary, indent=2))


def run_batch(args):
    import requests
    key = Path(args.key_file).expanduser().resolve()
    if not key.is_file():
        raise ValueError('Create your SnapJudge key file first; see README. Do not paste the key in chat.')
    if not key.read_text().strip().startswith('sjk_'):
        raise ValueError('The key file must contain the separately issued sjk_ key.')
    output = export(args.run, allow_partial=getattr(args, 'allow_partial', False))
    if not any((output / 'documents').glob('*.txt')):
        raise ValueError('No eligible answers of at least 50 words; see coverage.json.')
    client = Path('.cache/snapjudge/snapjudge_batch.py').resolve()
    if not client.exists():
        response = requests.get(CLIENT_URL, timeout=(15, 60))
        response.raise_for_status()
        source = response.content
        compile(source, CLIENT_URL, 'exec')
        client.parent.mkdir(parents=True, exist_ok=True)
        temporary = client.with_suffix('.tmp')
        temporary.write_bytes(source)
        temporary.replace(client)
    write_json(output / 'client.json', {'source': CLIENT_URL, 'sha256': digest(client),
                                       'save_submissions': False, 'concurrency': args.concurrency})
    result = subprocess.run([sys.executable, str(client), '--key', str(key),
                             '--input', str(output / 'documents'),
                             '--out', str(output / 'results.csv'),
                             '--json-dir', str(output / 'results_json'),
                             '--no-save', '--concurrency', str(args.concurrency)])
    if (output / 'results.csv').exists():
        report(args.run)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('export', 'run', 'report'):
        command = commands.add_parser(name)
        command.add_argument('--run', default='runs/sas-balanced')
        command.add_argument('--allow-partial', action='store_true',
                             help='Match saved human/AI counts per question without more generation')
        if name == 'run':
            command.add_argument('--key-file', default='.secrets/snapjudge.key')
            command.add_argument('--concurrency', type=int, choices=range(1, 11), default=3)
    args = parser.parse_args()
    if args.command == 'export':
        export(args.run, allow_partial=args.allow_partial)
    elif args.command == 'report':
        report(args.run)
    else:
        sys.exit(run_batch(args))


if __name__ == '__main__':
    main()
