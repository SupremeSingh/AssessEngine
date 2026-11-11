"""Pinned ASAP-SAS science/biology preparation; no model dependencies."""
from collections import Counter
import hashlib
import random

DATASET = 'nlpatunt/D_ASAP-SAS'
REVISION = 'f97ad6415c94c763b901423b72f78f1fbf46d630'
SETS = (1, 2, 5, 6, 10)


def clean_prompt(text, essay_set):
    if essay_set in (5, 6):
        marker = '\nRubric for '
        if text.count(marker) != 1:
            raise ValueError(f'Expected one rubric boundary in set {essay_set}')
        text = text.split(marker)[0]
        text = text.removeprefix('Prompt—Protein Synthesis Item\n')
    if any(term in text.casefold() for term in ('rubric', 'key elements:')):
        raise ValueError(f'Unexpected grading material in set {essay_set}')
    if not text.strip():
        raise ValueError('Empty prompt')
    return text.strip()


def prepare_records(rows, humans_per_question=50, seed=42):
    groups = {i: [] for i in SETS}
    for row in rows:
        number = int(row['essay_set'])
        if number in groups:
            groups[number].append(row)
    questions, coverage = [], []
    for number, group in groups.items():
        if not group:
            raise ValueError(f'Missing SAS essay set {number}')
        prompts = {r['prompt'] for r in group}
        if len(prompts) != 1:
            raise ValueError(f'Inconsistent prompts in set {number}')
        question = clean_prompt(prompts.pop(), number)
        unique = {}
        for row in sorted(group, key=lambda r: (r['_split'], str(r['Id']))):
            text = row['essay_text']
            if not isinstance(text, str):
                raise ValueError('Expected raw human answer text')
            # Exact text duplicates within a prompt cannot inflate response counts.
            unique.setdefault(text, row)
        eligible = [r for text, r in unique.items() if len(text.split()) >= 50]
        eligible.sort(key=lambda r: str(r['Id']))
        if len(eligible) < humans_per_question:
            raise ValueError(f'Set {number}: only {len(eligible)} eligible answers; '
                             f'requested {humans_per_question}')
        selected = random.Random(seed + number).sample(eligible, humans_per_question)
        humans = []
        for row in selected:
            text = row['essay_text']
            humans.append({'id': f"sas-{number}-{row['Id']}", 'student_answer': text,
                           'source_id': row['Id'], 'source_split': row['_split'],
                           'words': len(text.split())})
        questions.append({
            'question_id': hashlib.sha256(f'asap-sas:{number}'.encode()).hexdigest()[:16],
            'essay_set': number, 'question': question, 'humans': humans,
            'source_prompt_sha256': hashlib.sha256(group[0]['prompt'].encode()).hexdigest(),
            'source_prompt_url': group[0]['prompt_&_source_essay_urls'],
            'source_image_url': group[0].get('image_urls'),
            'prompt_processing': ('Removed rubric and item title' if number == 5 else
                                  'Removed rubric' if number == 6 else
                                  'Mirror text including procedures and data tables; whitespace trimmed'),
        })
        coverage.append({'essay_set': number, 'loaded_answers': len(group),
                         'unique_answers': len(unique), 'eligible_answers': len(eligible),
                         'selected_answers': len(humans),
                         'source_splits': dict(Counter(r['_split'] for r in group))})
    return questions, coverage


def response_prompt(question, min_words=50, max_words=100):
    if min_words < 50 or max_words < min_words:
        raise ValueError('Use 50 <= min_words <= max_words')
    return (question['question'] + '\n\n'
            f'Write your final answer in {min_words}–{max_words} words.')
