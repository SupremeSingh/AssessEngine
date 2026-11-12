"""Budgeted, sequential Pangram realtime scoring. `plan` makes no API calls."""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .core import digest, read_jsonl, write_json, write_jsonl
from .generation import qualifies

API = 'https://text.external-api.pangram.com'
MODEL = 'pangram-4'
USD_PER_CREDIT = Decimal('0.05')


def save_ledger(path, ledger):
    write_json(path, ledger)
    # Flush the task reservation and directory entry before any billable POST.
    with path.open('rb') as stream:
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def credit_estimate(text):
    return max(1, math.ceil(len(text.split()) / 100))


def inputs(run, allow_partial=False):
    run = Path(run)
    coverage = json.loads((run / 'generation_coverage.json').read_text())
    if not coverage['all_quotas_met'] and not allow_partial:
        raise ValueError('Generation quotas are incomplete. Use --allow-partial to select balanced saved answers.')
    rows = []
    answers = read_jsonl(run / 'answers.jsonl')
    for q in read_jsonl(run / 'questions.jsonl'):
        ai = sorted([r for r in answers if r['question_id'] == q['question_id'] and qualifies(r)],
                    key=lambda r: (r['sample'], r.get('attempt', 1), r['id']))
        if len({r['sample'] for r in ai}) != len(ai):
            raise ValueError('Multiple accepted attempts for one slot; inspect saved generation data.')
        humans = q['humans']
        if allow_partial:
            count = min(len(humans), len(ai))
            # Original deterministic human sample order; never select by detector score.
            humans, ai = humans[:count], ai[:count]
        rows.extend({'id': h['id'], 'question_id': q['question_id'], 'authorship': 'human',
                     'text': h['student_answer']} for h in humans)
        rows.extend({'id': r['id'], 'question_id': r['question_id'], 'authorship': 'ai',
                     'text': r['answer']} for r in ai)
    if not rows:
        raise ValueError('No qualifying matched answers are available.')
    for row in rows:
        if len(row['text'].split()) < 50:
            raise ValueError('Unexpected short input; no text will be padded or rewritten.')
        row['words'] = len(row['text'].split())
        row['credits_estimate'] = credit_estimate(row['text'])
        row['cache_key'] = hashlib.sha256((MODEL + '\0' + row['text']).encode()).hexdigest()
    return rows


def make_plan(rows, ledger, max_credits=1500, total_credits=3000, reserve_credits=1500):
    unique = {r['cache_key']: r for r in rows}
    new = {key: row for key, row in unique.items() if key not in ledger}
    estimated = sum(r['credits_estimate'] for r in unique.values())
    additional = sum(r['credits_estimate'] for r in new.values())
    reserved = sum(e['credits_estimate'] for e in ledger.values())
    allowed = estimated <= max_credits and reserved + additional <= total_credits - reserve_credits
    return {'model': MODEL, 'documents': len(rows), 'by_authorship': dict(Counter(r['authorship'] for r in rows)),
            'by_question': {qid: dict(Counter(r['authorship'] for r in rows
                                             if r.get('question_id', 'unknown') == qid))
                            for qid in sorted({r.get('question_id', 'unknown') for r in rows})},
            'unique_texts': len(unique), 'estimated_run_credits': estimated,
            'estimated_new_credits': additional, 'ledger_reserved_credits': reserved,
            'max_run_credits': max_credits, 'total_budget_credits': total_credits,
            'reserve_credits': reserve_credits, 'within_budget': allowed,
            'billing_assumption': 'One credit per started 100 whitespace-counted words, at $0.05. '
                                  'Estimate only; provider counts and account billing are authoritative.'}


class PangramHTTPError(RuntimeError):
    def __init__(self, status, method, path):
        self.status, self.method, self.path = status, method, path
        super().__init__(f'Pangram HTTP {status} on {method} {path}; '
                         'stopped without automatically repeating a paid submission.')


class Client:
    def __init__(self, key):
        import requests
        self.session = requests.Session()
        self.session.headers.update({'x-api-key': key})
        # Avoid retry adapters: a POST timeout can represent a billable accepted task.

    def request(self, method, path, **kwargs):
        response = self.session.request(method, API + path, timeout=(15, 60), **kwargs)
        if not response.ok:
            raise PangramHTTPError(response.status_code, method, path)
        return response.json()

    def balance(self):
        return Decimal(self.request('GET', '/balance')['available_balance_usd'])


def process(row, ledger, save, client, reserve_credits=1500, poll_seconds=3, poll_timeout=1800,
            allow_missing_balance=False):
    key = row['cache_key']
    entry = ledger.get(key)
    if entry is None:
        available = None
        if not (allow_missing_balance and getattr(client, 'balance_unavailable', False)):
            try:
                available = client.balance()
            except PangramHTTPError as error:
                if not (allow_missing_balance and error.status == 404 and
                        error.method == 'GET' and error.path == '/balance'):
                    raise
                client.balance_unavailable = True
                print('Balance endpoint returned 404: using the explicitly enabled local estimated-credit '
                      'ledger only. Live account reserve cannot be verified.', flush=True)
        # Live-balance check includes a 10% cushion on the next estimated charge.
        needed = (Decimal(reserve_credits) + Decimal(row['credits_estimate']) * Decimal('1.1')) * USD_PER_CREDIT
        if available is not None and available < needed:
            raise RuntimeError('Live API balance is too low to preserve the configured reserve. '
                               'Check that your 3,000 credits are API credits at the documented rate.')
        entry = {'state': 'submitting', 'credits_estimate': row['credits_estimate'],
                 'balance_guard': 'live' if available is not None else 'unavailable_404'}
        ledger[key] = entry
        save()  # Durable reservation before the POST, including ambiguous outcomes.
        response = client.request('POST', '/task', json={
            'model': MODEL, 'text': row['text'], 'public_dashboard_link': False})
        task_id = response.get('task_id')
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError('Submission returned no task_id. Reconcile this submission before retrying.')
        entry.update(state='polling', task_id=task_id)
        save()
    if entry['state'] == 'submitting':
        raise RuntimeError('Uncertain previous submission: no saved task ID. '
                           'Reconcile it with Pangram before retrying; automatic resubmission is blocked.')
    started = time.monotonic()
    while entry['state'] == 'polling':
        result = client.request('GET', '/task/' + entry['task_id'])
        stage = result.get('stage')
        if stage in ('STAGE_SUCCESS', 'STAGE_FAILED'):
            entry.update(state='done' if stage == 'STAGE_SUCCESS' else 'failed', result=result)
            save()
            break
        if time.monotonic() - started >= poll_timeout:
            raise TimeoutError('Polling timed out. Resume the same command to poll the saved task; no new POST.')
        time.sleep(poll_seconds)
    return entry


def report(run, rows, ledger):
    output = Path(run) / 'pangram'
    joined = []
    for row in rows:
        entry = ledger.get(row['cache_key'], {})
        joined.append({**row, 'task_id': entry.get('task_id'), 'state': entry.get('state', 'not_submitted'),
                       'result': entry.get('result')})
    write_jsonl(output / 'scores.jsonl', joined)
    with (output / 'scores.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['id', 'question_id', 'authorship', 'words', 'task_id', 'state',
                  'version', 'prediction_short', 'fraction_ai', 'fraction_ai_assisted', 'fraction_human']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in joined:
            flat = {key: row.get(key) for key in fields}
            flat.update({key: (row['result'] or {}).get(key) for key in fields[6:]})
            writer.writerow(flat)
    summary = {'detector': MODEL, 'note': 'Text fractions are not authorship probabilities. '
               'AI sample is conditioned on successful generation of 50+ words; rejected attempts remain in answers.jsonl.'}
    for label in ('human', 'ai'):
        group = [r for r in joined if r['authorship'] == label]
        completed = [r['result'] for r in group if r['state'] == 'done']
        summary[label] = {'total': len(group), 'completed': len(completed),
                          'states': dict(Counter(r['state'] for r in group)),
                          'verdicts': dict(Counter(r['prediction_short'] for r in completed)),
                          'versions': dict(Counter(r.get('version', 'unknown') for r in completed))}
        for field in ('fraction_ai', 'fraction_ai_assisted', 'fraction_human'):
            values = [float(r[field]) for r in completed]
            summary[label]['mean_' + field] = sum(values) / len(values) if values else None
    write_json(output / 'report.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run', 'report'])
    parser.add_argument('--run', default='runs/sas-balanced')
    parser.add_argument('--ledger', default='.cache/pangram/ledger.json')
    parser.add_argument('--max-credits', type=int, default=1500)
    parser.add_argument('--total-credits', type=int, default=3000)
    parser.add_argument('--reserve-credits', type=int, default=1500)
    parser.add_argument('--allow-partial', action='store_true',
                        help='Match saved qualifying AI/human counts within each question, without generation')
    parser.add_argument('--allow-missing-balance', action='store_true',
                        help='Permit local estimated-credit limits if GET /balance returns 404 only')
    args = parser.parse_args()
    if min(args.max_credits, args.total_credits) <= 0 or not 0 <= args.reserve_credits < args.total_credits:
        parser.error('Invalid credit budget')
    rows = inputs(args.run, args.allow_partial)
    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    # All jobs sharing this ledger must serialize; no simultaneous paid submissions.
    with ledger_path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        plan = make_plan(rows, ledger, args.max_credits, args.total_credits, args.reserve_credits)
        plan['selection_policy'] = ('balanced_saved_answers_v1' if args.allow_partial else 'full_quota_v1')
        plan['allow_missing_balance'] = args.allow_missing_balance
        output = Path(args.run) / 'pangram'
        write_json(output / 'plan.json', plan)
        write_jsonl(output / 'selection.jsonl', [
            {k: r.get(k) for k in ('id', 'question_id', 'authorship', 'words', 'cache_key')} for r in rows])
        print(json.dumps(plan, indent=2), flush=True)
        if args.command == 'plan':
            return
        if args.command == 'report':
            report(args.run, rows, ledger)
            return
        if not plan['within_budget']:
            raise RuntimeError('Budget exceeded. Nothing submitted. Inspect plan.json before changing limits.')
        key = os.environ.get('PANGRAM_API_KEY', '').strip()
        if not key:
            raise ValueError('Set PANGRAM_API_KEY in .env.pangram; see config/pangram.env.example.')
        client = Client(key)
        if MODEL not in client.request('GET', '/models')['models']:
            raise RuntimeError('Your key does not offer pangram-4. No inference submitted.')
        fingerprint = {'questions_sha256': digest(Path(args.run) / 'questions.jsonl'),
                       'answers_sha256': digest(Path(args.run) / 'answers.jsonl'), 'model': MODEL,
                       'selection_policy': plan['selection_policy'],
                       'selection_sha256': digest(output / 'selection.jsonl')}
        old = output / 'inputs.json'
        if old.exists() and json.loads(old.read_text()) != fingerprint:
            raise ValueError('Run inputs changed. Use a separate run directory.')
        write_json(old, fingerprint)
        try:
            for index, row in enumerate(rows, 1):
                entry = process(row, ledger, lambda: save_ledger(ledger_path, ledger), client,
                                args.reserve_credits, allow_missing_balance=args.allow_missing_balance)
                print(f"{index}/{len(rows)} {row['id']}: {entry['state']}", flush=True)
                if entry['state'] == 'failed':
                    raise RuntimeError('Pangram task failed; saved response for inspection. No automatic paid retry.')
        finally:
            report(args.run, rows, ledger)


if __name__ == '__main__':
    main()
