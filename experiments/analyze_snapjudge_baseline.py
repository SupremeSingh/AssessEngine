"""Recompute descriptive baseline metrics from the supplied archive; no extraction/API calls."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import mean, median
import tarfile


def lengths(rows):
    values = [len(r['text'].split()) for r in rows]
    return dict(n=len(values), total_words=sum(values), minimum=min(values),
                mean=mean(values), median=median(values), maximum=max(values))


def confusion(rows):
    tp = sum(r['authorship'] == 'ai' and r['predicted_ai'] for r in rows)
    fn = sum(r['authorship'] == 'ai' and not r['predicted_ai'] for r in rows)
    fp = sum(r['authorship'] == 'human' and r['predicted_ai'] for r in rows)
    tn = sum(r['authorship'] == 'human' and not r['predicted_ai'] for r in rows)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, accuracy=(tp+tn)/len(rows),
                ai_recall=tp/(tp+fn), human_specificity=tn/(tn+fp),
                ai_precision=tp/(tp+fp), ai_f1=2*tp/(2*tp+fp+fn))


def analyze(archive):
    root = 'runs/sas-balanced/'
    with tarfile.open(archive) as bundle:
        def data(path):
            return bundle.extractfile(path).read()

        def json_file(path):
            return json.loads(data(path))

        def jsonl(path):
            return [json.loads(line) for line in data(path).splitlines()]

        questions = jsonl(root+'questions.jsonl')
        attempts = jsonl(root+'answers.jsonl')
        index = jsonl(root+'snapjudge/index.jsonl')
        csv_rows = list(csv.DictReader(io.StringIO(data(root+'snapjudge/results.csv').decode('utf-8-sig'))))
        by_file = {row['file']: row for row in csv_rows}
        if len(by_file) != len(csv_rows):
            raise ValueError('Duplicate result file')
        rows = []
        raw_results = []
        sources = {h['id']: h['student_answer'] for q in questions for h in q['humans']}
        sources.update({a['id']: a['answer'] for a in attempts})
        for row in index:
            if row['excluded_reason'] is not None:
                continue
            result = by_file[row['file']]
            if result['error'] or result['verdict'] not in ('AI-generated', 'Human-written'):
                raise ValueError('Unexpected result/error in this binary-verdict baseline')
            raw = json_file(root+'snapjudge/results_json/'+row['file']+'.json')
            document = data(root+'snapjudge/documents/'+row['file']).decode('utf-8')
            if document != row['text'] or row['text'] != sources[row['id']]:
                raise ValueError('Exported answer differs from source')
            if raw['saved_for_research'] is not False or raw['file'] != row['file']:
                raise ValueError('Unexpected raw-result metadata')
            if raw['answer']['results'][0]['generated_fraction'] != float(result['ai_generated_share']):
                raise ValueError('Raw and CSV scores disagree')
            rows.append(dict(row, predicted_ai=result['verdict']=='AI-generated',
                             raw_tier=raw['answer']['results'][0]['tier']))
            raw_results.append(raw)
        if len(rows) != len(csv_rows):
            raise ValueError('Index/result counts differ')
        manifest = json_file(root+'snapjudge/export.json')
        for filename, field in [('questions.jsonl','questions_sha256'),('answers.jsonl','answers_sha256')]:
            if hashlib.sha256(data(root+filename)).hexdigest() != manifest[field]:
                raise ValueError('Source hash mismatch')
        per_question = []
        for question in questions:
            group = [r for r in rows if r['question_id']==question['question_id']]
            generated = [r for r in attempts if r['question_id']==question['question_id']]
            per_question.append(dict(essay_set=question['essay_set'], question_id=question['question_id'],
                metrics=confusion(group), human_lengths=lengths([r for r in group if r['authorship']=='human']),
                ai_lengths=lengths([r for r in group if r['authorship']=='ai']), attempts=len(generated),
                rejected=sum(not r['accepted'] for r in generated)))
        misses = [dict(id=r['id'], essay_set=next(q['essay_set'] for q in questions if q['question_id']==r['question_id']),
                       words=r['words'], raw_tier=r['raw_tier'])
                  for r in rows if r['authorship']=='ai' and not r['predicted_ai']]
        bins = []
        for low, high, label in [(50,79,'50–79'),(80,120,'80–120'),(121,100000,'121+')]:
            group = [r for r in rows if low<=r['words']<=high]
            bins.append(dict(words=label, metrics=confusion(group)))
        metrics = confusion(rows)
        return dict(archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
            metrics=metrics, questions=per_question, misses=misses, length_bins=bins,
            raw_tiers={label:dict(Counter(r['raw_tier'] for r in rows if r['authorship']==label))
                       for label in ['human','ai']},
            human_lengths=lengths([r for r in rows if r['authorship']=='human']),
            ai_lengths=lengths([r for r in rows if r['authorship']=='ai']),
            missed_ai_lengths=lengths([r for r in rows if r['authorship']=='ai' and not r['predicted_ai']]),
            detected_ai_lengths=lengths([r for r in rows if r['authorship']=='ai' and r['predicted_ai']]),
            attempts=len(attempts), accepted=sum(r['accepted'] for r in attempts),
            rejected=sum(not r['accepted'] for r in attempts),
            statuses=dict(Counter(r['status'] for r in attempts)),
            generated_tokens=sum(r['generated_tokens'] for r in attempts),
            generation_seconds=sum(r['seconds'] for r in attempts),
            gpus=dict(Counter(r['gpu'] for r in attempts)),
            windows=dict(Counter(r['answer']['results'][0]['windows'] for r in raw_results)),
            analysis_timestamps=[min(r['analysed_at'] for r in raw_results),max(r['analysed_at'] for r in raw_results)],
            sum_client_document_seconds=sum(r['seconds'] for r in raw_results),
            detector_metadata={k:raw_results[0]['answer']['results'][0][k]
                for k in ['model','classes','layer','thr_nonhuman','thr_generated','score_type']},
            generation_manifest=json_file(root+'generation.json'), dataset_manifest=json_file('data/sas/dataset.json'),
            dataset_coverage=json_file('data/sas/coverage.json'),
            client=json_file(root+'snapjudge/client.json'), export=manifest,
            reporting_commit=data(root+'report_git_commit.txt').decode().strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', nargs='?', default='sas-report-data.tar.gz')
    parser.add_argument('--out', default='experiments/snapjudge-sas-baseline.metrics.json')
    args = parser.parse_args()
    result = analyze(args.archive)
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['metrics','human_lengths','ai_lengths','length_bins']}, indent=2))
