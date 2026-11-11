"""Count full-dataset answer lengths without loading models or calling detectors."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median

from .core import write_json

DATASET = 'nkazi/MohlerASAG'


def summarize(rows, minimum=50):
    groups = defaultdict(list)
    for row in rows:
        key = ' '.join(row['question'].split()).casefold()
        groups[key].append(row)
    questions = []
    lengths = []
    for key, group in sorted(groups.items()):
        counts = [len(row['student_answer'].split()) for row in group]
        lengths.extend(counts)
        eligible = sum(n >= minimum for n in counts)
        questions.append({
            'question': group[0]['question'],
            'source_question_ids': ','.join(sorted({r['id'].rsplit('.', 1)[0] for r in group})),
            'human_answers': len(group), 'eligible_answers': eligible,
            'eligible_fraction': eligible / len(group),
            'median_words': median(counts), 'max_words': max(counts),
        })
    total = len(lengths)
    eligible = sum(n >= minimum for n in lengths)
    summary = {
        'minimum_words': minimum, 'word_count_method': 'Whitespace-separated tokens; service may tokenize differently.',
        'human_answers': total, 'distinct_question_wordings': len(questions),
        'eligible_answers': eligible,
        'eligible_percent': round(100 * eligible / total, 2) if total else 0,
        'median_words': median(lengths) if total else None,
        'questions_with_eligible_answers': {
            str(n): sum(q['eligible_answers'] >= n for q in questions)
            for n in (1, 3, 5, 10, 20)
        },
        'answer_length_counts': {
            '0': sum(n == 0 for n in lengths),
            '1-19': sum(1 <= n < 20 for n in lengths),
            '20-49': sum(20 <= n < 50 for n in lengths),
            '50-99': sum(50 <= n < 100 for n in lengths),
            '100-199': sum(100 <= n < 200 for n in lengths),
            '200+': sum(n >= 200 for n in lengths),
        },
        'limitation': 'Length eligibility only; not answer validity, independence, or educational suitability.',
    }
    return summary, sorted(questions, key=lambda q: (-q['eligible_answers'], q['question']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['sas', 'mohler'], default='sas')
    parser.add_argument('--out')
    parser.add_argument('--revision', help='Dataset commit; defaults to the previously recorded commit or latest')
    args = parser.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi

    output = Path(args.out or f'data/diagnostics/{args.dataset}')
    report_path = output / 'report.json'
    saved = json.loads(report_path.read_text()) if report_path.exists() else {}
    sha = args.revision or saved.get('dataset_revision')
    if args.dataset == 'sas':
        from .sas import DATASET as sas_dataset, REVISION
        repo = sas_dataset
        sha = sha or REVISION
        loaded = load_dataset(repo, revision=sha)
        seen, dataset = set(), []
        for part in loaded.values():
            for row in part:
                identity = (row['essay_set'], row['essay_text'])
                if identity in seen:
                    continue
                seen.add(identity)
                dataset.append({'id': f"SAS{row['essay_set']}.A{row['Id']}",
                                'question': row['prompt'], 'student_answer': row['essay_text']})
        subset, split = 'default (exact duplicates removed within prompts)', 'all'
    else:
        repo = DATASET
        sha = sha or HfApi().dataset_info(repo).sha
        dataset = load_dataset(repo, name='raw', split='open_ended', revision=sha)
        subset, split = 'raw', 'open_ended'
    summary, questions = summarize(dataset)
    summary.update(dataset=repo, dataset_revision=sha, subset=subset, split=split)
    write_json(report_path, summary)
    with (output / 'questions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(questions[0]) if questions else ['question'])
        writer.writeheader()
        writer.writerows(questions)
    print(json.dumps(summary, indent=2))
    print(f'Per-question breakdown: {output / "questions.csv"}')


if __name__ == '__main__':
    main()
