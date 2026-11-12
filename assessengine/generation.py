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
        target = samples[question['question_id']] if isinstance(samples, dict) else samples
        group = [r for r in rows if r['question_id'] == question['question_id']]
        accepted = len({r['sample'] for r in group if qualifies(r)})
        groups.append({'question_id': question['question_id'], 'target': target,
                       'accepted': accepted, 'missing': max(0, target - accepted),
                       'attempts': len(group), 'rejected': sum(not qualifies(r) for r in group)})
    return {'minimum_final_answer_words': 50, 'max_attempts_per_slot': max_attempts,
            'all_quotas_met': all(g['missing'] == 0 for g in groups), 'questions': groups}


def validate_resume_source(previous, current, questions, rows):
    """Reuse attempts only when inference settings match and quotas do not shrink."""
    quota_keys = {'samples', 'samples_by_question'}
    if {k:v for k,v in previous.items() if k not in quota_keys} != {
            k:v for k,v in current.items() if k not in quota_keys}:
        raise ValueError('Resume source has different generation settings or question inputs')
    old = previous.get('samples_by_question', {})
    new = current.get('samples_by_question', {})
    for q in questions:
        qid = q['question_id']
        if new.get(qid, current['samples']) < old.get(qid, previous['samples']):
            raise ValueError('Resume quotas must not shrink')
    valid_ids = {q['question_id'] for q in questions}
    if any(r['question_id'] not in valid_ids for r in rows):
        raise ValueError('Unknown question in saved answers')
