"""Minimal-instruction, one-shot Gemini rewriting; separate from the paper-guided protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from .core import digest, read_jsonl, write_json, write_jsonl
from .gemini import Client, ROOT, cached_call

SCHEMA = {'type':'object','properties':{k:{'type':'string'} for k in
          ['rewritten_question','rationale','meaning_preservation_note']},
          'required':['rewritten_question','rationale','meaning_preservation_note'],
          'additionalProperties':False}


def prepare(args):
    source = Path(args.data)/'questions.jsonl'
    questions = read_jsonl(source)
    if sorted(q['essay_set'] for q in questions)!=[1,2,5,6,10]:
        raise ValueError('Expected the five SAS science/biology prompts')
    instruction = (ROOT/'prompts/exploratory-rewriter.md').read_text()
    output = Path(args.out)
    config = {'project':json.loads((ROOT/'config/gemini.project.json').read_text()),
              'model':args.model,'questions_sha256':digest(source),
              'instruction_sha256':hashlib.sha256(instruction.encode()).hexdigest(),
              'protocol':'exploratory_minimal_v1','one_shot':True,'detector_feedback':False,
              'word_limit':None,'paper_context_supplied':False}
    path = output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError('Configuration changed; use a new output directory')
    write_json(path,config)
    records = [{k:q[k] for k in ['question_id','essay_set','question']} for q in questions]
    write_jsonl(output/'questions.jsonl',records)
    for q in records:
        request = {'systemInstruction':{'parts':[{'text':instruction}]},
                   'contents':[{'role':'user','parts':[{'text':q['question']}]}],
                   'generationConfig':{'responseMimeType':'application/json','responseJsonSchema':SCHEMA}}
        # No maxOutputTokens or word-count instruction; provider limits still apply.
        write_json(output/'requests'/f"{q['question_id']}.json",request)
    print(f'Prepared {len(records)} minimal requests; no API calls.')


def generate(args):
    output = Path(args.out)
    manifest = json.loads((output/'manifest.json').read_text())
    client = Client(manifest['model'])
    rows = []
    for q in read_jsonl(output/'questions.jsonl'):
        qid = q['question_id']
        request = json.loads((output/'requests'/f'{qid}.json').read_text())
        proposal = cached_call(client,output/'calls'/f'{qid}-proposal',request)
        if not all(isinstance(proposal.get(k),str) and proposal[k].strip() for k in SCHEMA['required']):
            raise ValueError('Incomplete Gemini proposal; inspect saved response')
        if proposal['rewritten_question'].strip()==q['question'].strip():
            raise ValueError('Gemini returned an unchanged question; no automatic retry')
        rows.append({**q,**proposal,'semantic_status':'Unverified; Gemini statement is not independent validation'})
        write_jsonl(output/'proposals.jsonl',rows)
        print(f"Set {q['essay_set']}: rewrite saved; no equivalence claim validated",flush=True)
    page = '# Exploratory Gemini rewrites\n\nOne call per question; no word limit, paper brief, or feedback.\n'
    for q in rows:
        page+=f"\n## Set {q['essay_set']}\n\n### Original\n\n{q['question']}\n\n"
        page+=f"### Gemini rewrite\n\n{q['rewritten_question']}\n\n"
        page+=f"Intended effect: {q['rationale']}\n\nMeaning note: {q['meaning_preservation_note']}\n\n"
        page+=f"Assessment status: {q['semantic_status']}\n"
    (output/'review.md').write_text(page)
    write_json(output/'run-summary.json',{'rewrites':len(rows),'successful_calls':len(rows),
        'model':manifest['model'],'project':manifest['project']['project_id'],
        'protocol':manifest['protocol'],'meaning_equivalence_verified':False,
        'answer_correctness_or_detector_effect_measured':False})


def materialize(args):
    """Freeze exact Gemini outputs for a separate generation run, with no human labels."""
    source = Path(args.proposals)
    rows = read_jsonl(source)
    if sorted(row['essay_set'] for row in rows) != [1, 2, 5, 6, 10]:
        raise ValueError('Expected exactly the five exploratory rewrites')
    if len({row['question_id'] for row in rows}) != 5:
        raise ValueError('Duplicate question IDs')
    records = []
    for row in rows:
        rewritten = row['rewritten_question']
        if not isinstance(rewritten, str) or not rewritten.strip():
            raise ValueError('Empty rewritten question')
        records.append({'question_id': row['question_id'], 'essay_set': row['essay_set'],
                        'question': rewritten, 'original_question': row['question'],
                        'condition': 'exploratory_minimal_v1', 'humans': [],
                        'semantic_status': 'Unverified', 'proposals_sha256': digest(source)})
    destination = Path(args.data)/'questions.jsonl'
    if destination.exists() and read_jsonl(destination) != records:
        raise ValueError('Prepared inputs changed; choose a new data directory')
    write_jsonl(destination, records)
    print(f'Prepared {len(records)} exact Gemini rewrites; no historical human answers included.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','generate','materialize'])
    parser.add_argument('--data',default='data/sas')
    parser.add_argument('--out',default='runs/gemini-exploratory')
    parser.add_argument('--model',default=os.environ.get('GEMINI_MODEL','gemini-3.1-pro-preview'))
    parser.add_argument('--proposals',default='experiments/inputs/gemini-exploratory-v1.jsonl')
    args = parser.parse_args()
    globals()[args.command](args)


if __name__=='__main__':
    main()
