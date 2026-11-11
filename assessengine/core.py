"""Dependency-free data handling and response validation."""
import hashlib
import json
import os
import random
from pathlib import Path


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def append_jsonl(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def question_records(rows, count, seed):
    """Sample distinct question wordings, preserving all raw human responses."""
    groups = {}
    for row in rows:
        # Duplicate wording across assignments must not become separate questions.
        key = ' '.join(row['question'].split()).casefold()
        if key not in groups:
            groups[key] = {'question': row['question'], 'humans': []}
        groups[key]['humans'].append(row)
    keys = sorted(groups)
    if not 1 <= count <= len(keys):
        raise ValueError(f'Choose 1..{len(keys)} distinct questions; got {count}')
    selected = random.Random(seed).sample(keys, count)
    return [dict(question_id=hashlib.sha256(key.encode()).hexdigest()[:16],
                 **groups[key]) for key in selected]


def split_response(text, ended):
    """The chat template may supply <think>; a closing tag is still required.

    Fail closed: never score unfinished reasoning as an answer.
    """
    if '</think>' not in text:
        return {'reasoning': text, 'answer': '', 'status': 'missing_reasoning_boundary'}
    reasoning, answer = text.split('</think>', 1)
    reasoning = reasoning.removeprefix('<think>').strip()
    answer = answer.strip()
    status = 'complete' if ended and answer else ('empty_answer' if not answer else 'truncated')
    return {'reasoning': reasoning, 'answer': answer, 'status': status}
