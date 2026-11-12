"""Select saved answers without model inference or detector calls."""
import json
from pathlib import Path

from .core import read_jsonl
from .generation import qualifies


def select_answers(run, allow_partial=False):
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
            humans, ai = humans[:count], ai[:count]
        rows.extend({'id': h['id'], 'question_id': q['question_id'], 'authorship': 'human',
                     'text': h['student_answer']} for h in humans)
        rows.extend({'id': r['id'], 'question_id': r['question_id'], 'authorship': 'ai',
                     'text': r['answer']} for r in ai)
    if not rows:
        raise ValueError('No qualifying matched answers are available.')
    for row in rows:
        row['words'] = len(row['text'].split())
        if row['words'] < 50:
            raise ValueError('Unexpected short input; no text will be padded or rewritten.')
    return rows
