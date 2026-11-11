"""Quota and retry bookkeeping independent of model inference."""
import hashlib


def qualifies(row):
    return row['status'] == 'complete' and len(row['answer'].split()) >= 50


def next_attempt(rows, samples, max_attempts):
    """Return the next unfilled slot and unused attempt; exhausted slots stay failed."""
    ids = [(row['sample'], row['attempt']) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate generation attempt in saved answers')
    for sample in range(samples):
        saved = [row for row in rows if row['sample'] == sample]
        if any(qualifies(row) for row in saved):
            continue
        used = {row['attempt'] for row in saved}
        for attempt in range(1, max_attempts + 1):
            if attempt not in used:
                return sample, attempt
    return None


def attempt_seed(base, question_id, sample, attempt):
    value = f'{base}:{question_id}:{sample}:{attempt}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], 'big')


def quota_report(questions, rows, samples, max_attempts):
    groups = []
    for question in questions:
        group = [r for r in rows if r['question_id'] == question['question_id']]
        accepted = len({r['sample'] for r in group if qualifies(r)})
        groups.append({'question_id': question['question_id'], 'target': samples,
                       'accepted': accepted, 'missing': max(0, samples - accepted),
                       'attempts': len(group), 'rejected': sum(not qualifies(r) for r in group)})
    return {'minimum_final_answer_words': 50, 'max_attempts_per_slot': max_attempts,
            'all_quotas_met': all(g['missing'] == 0 for g in groups), 'questions': groups}
