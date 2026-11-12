"""One paper-informed Gemini rewrite per question and original-versus-modified export."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

from .core import digest, read_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
KINDS = ('paper_inspired',)
MARKERS = {1: 'After reading the group', 2: '1. Draw a conclusion',
           5: None, 6: None, 10: 'Prompt: Use the results'}
REQUIREMENTS = {
    1: 'Identify at least three missing pieces of information needed to replicate the supplied experiment.',
    2: 'Draw a conclusion from the supplied data AND describe two improvements to design and/or validity.',
    5: 'Starting with mRNA leaving the nucleus, list AND describe four major steps of protein synthesis.',
    6: 'List AND describe three processes controlling substance movement across the cell membrane.',
    10: 'Choose a listed doghouse paint color, describe its effect on the inside temperature, and support with experiment results.',
}


def schema(item_fields):
    return {'type': 'object', 'properties': {'variants': {'type': 'array', 'minItems': 1, 'maxItems': 1,
        'items': {'type': 'object', 'properties': item_fields, 'required': list(item_fields),
                  'additionalProperties': False}}}, 'required': ['variants'], 'additionalProperties': False}


PROPOSAL_SCHEMA = schema({**{k: {'type': 'string'} for k in
    ['modified_task', 'equivalence_rationale', 'feature_hypothesis', 'caveat', 'counterexample_check']},
    'task_contract': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1},
    'equivalence_claim': {'type': 'boolean'},
    'kind': {'type': 'string', 'enum': list(KINDS)}})


def split_question(question):
    number = question['essay_set']
    text = question['question']
    marker = MARKERS[number]
    if marker is None:
        context, task = '', text
    else:
        if text.count(marker) != 1:
            raise ValueError(f'Unexpected task boundary in set {number}')
        offset = text.index(marker)
        context, task = text[:offset], text[offset:]
    suffix = ''
    if number == 10:
        marker = '\nChoose a color:'
        if task.count(marker) != 1:
            raise ValueError('Missing set-10 color options')
        offset = task.index(marker)
        task, suffix = task[:offset], task[offset:]
    return dict(question_id=question['question_id'], essay_set=number,
                immutable_context=context, original_task=task, immutable_suffix=suffix,
                required_task=REQUIREMENTS[number])


def local_issues(original, modified):
    if not isinstance(modified, str) or not modified.strip():
        return ['Empty/non-text task']
    issues = []
    if modified.strip() == original.strip():
        issues.append('Unchanged task')
    if re.search(r'\b(ai|llm|gemini|deepseek|detector|authorship|human|artificial)\b', modified, re.I):
        issues.append('Respondent-facing authorship/style cue')
    if re.search(r'\b(words?|tokens?|paragraphs?|bullet|headings?|essay)\b', modified, re.I):
        issues.append('New length/format instruction')
    if len(modified) > len(original)*2.5:
        issues.append('Task expanded beyond 2.5 times original length')
    aliases = {'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}
    def numbers(text):
        text = re.sub(r'(?m)^\s*\d+[.)]\s*', '', text)
        found = re.findall(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+(?:\.\d+)?)\b', text.lower())
        return Counter(aliases.get(x,x) for x in found)
    if numbers(original) != numbers(modified):
        issues.append('Changed substantive numeric/quantifier tokens')
    if 'at least' in original.lower() and 'at least' not in modified.lower():
        issues.append('Removed at-least quantifier')
    if re.search(r'\bmight\b', original, re.I) and not re.search(r'\b(might|could|may)\b', modified, re.I):
        issues.append('Changed tentative scope to definite wording')
    return issues


def body(system, payload, output_schema):
    return {'systemInstruction': {'parts': [{'text': system}]},
            'contents': [{'role': 'user', 'parts': [{'text': json.dumps(payload, ensure_ascii=False)}]}],
            'generationConfig': {'temperature': 0.7, 'maxOutputTokens': 8192,
                                 'responseMimeType': 'application/json', 'responseJsonSchema': output_schema}}


def prepare(args):
    source = Path(args.data)/'questions.jsonl'
    questions = read_jsonl(source)
    if sorted(q['essay_set'] for q in questions) != sorted(MARKERS):
        raise ValueError('Expected all five SAS science/biology questions exactly once')
    context = (ROOT/'prompts/style-paper-context.md').read_text()
    instructions = (ROOT/'prompts/question-modifier.md').read_text()
    output = Path(args.out)
    config = {'project': json.loads((ROOT/'config/gemini.project.json').read_text()),
              'questions_sha256': digest(source), 'model': args.model,
              'paper_context_sha256': hashlib.sha256(context.encode()).hexdigest(),
              'instructions_sha256': hashlib.sha256(instructions.encode()).hexdigest(), 'format_version': 3,
              'one_shot': True, 'max_generation_calls': 5, 'detector_feedback': False}
    manifest = output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Inputs/model/instructions changed. Use a new --out directory.')
    write_json(manifest, config)
    records = [split_question(q) for q in questions]
    write_jsonl(output/'questions.jsonl', records)
    for record in records:
        request = body(instructions+'\n\n'+context, record, PROPOSAL_SCHEMA)
        write_json(output/'requests'/f"{record['question_id']}.json", request)
        # Also usable as a copy/paste prompt in Google AI Studio, without student data.
        page = instructions+'\n\n'+context+'\n\n## Question input\n```json\n'+json.dumps(record,indent=2)+'\n```\n'
        path = output/'ai-studio'/f"set-{record['essay_set']}.md"
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(page)
    print(f'Prepared {len(records)} question requests under {output}; no API calls made.')


class GeminiHTTPError(RuntimeError):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(f'Gemini generation HTTP {status}; inspect the saved failure details. No automatic retry.')


class Client:
    def __init__(self, model):
        import requests
        key = os.environ.get('GEMINI_API_KEY','').strip()
        if not key:
            raise ValueError('Set GEMINI_API_KEY using config/gemini.env.example. Project IDs are not credentials.')
        if not re.fullmatch(r'[a-zA-Z0-9_.-]+',model):
            raise ValueError('Use a model ID without the models/ prefix')
        self.session = requests.Session()
        self.session.headers['x-goog-api-key'] = key
        self.url = 'https://generativelanguage.googleapis.com/v1beta/models/'+model
        response = self.session.get(self.url, timeout=(15,60))
        if not response.ok:
            raise RuntimeError(f'Gemini model lookup HTTP {response.status_code}; check key/project/model availability.')
        if 'generateContent' not in response.json().get('supportedGenerationMethods',[]):
            raise ValueError('Selected model does not support generateContent')

    def generate(self, request):
        response = self.session.post(self.url+':generateContent',json=request,timeout=(15,180))
        if not response.ok:
            detail = response.text.replace(self.session.headers['x-goog-api-key'],'[REDACTED]')
            raise GeminiHTTPError(response.status_code,detail)
        return response.json()


def parsed_response(response):
    candidates = response.get('candidates',[])
    if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
        raise ValueError('Gemini response incomplete or blocked; inspect saved response')
    parts = candidates[0]['content']['parts']
    return json.loads(''.join(p.get('text','') for p in parts if not p.get('thought',False)))


def cached_call(client, path, request):
    raw = path.with_suffix('.response.json')
    req = path.with_suffix('.request.json')
    pending = path.with_suffix('.pending.json')
    if req.exists() and json.loads(req.read_text()) != request:
        raise ValueError('Stored API request changed; use a new output directory')
    if raw.exists():
        return parsed_response(json.loads(raw.read_text()))
    if pending.exists():
        raise RuntimeError(f'Previous call has an uncertain outcome: {pending}. Inspect before retrying.')
    write_json(req,request)
    write_json(pending,{'state':'submitted_or_uncertain'})
    try:
        response = client.generate(request)
    except GeminiHTTPError as error:
        write_json(path.with_suffix('.failure.json'),{'http_status':error.status,'detail':error.detail})
        if error.status in (400,401,403,404,422,429):
            pending.unlink()  # Explicit rejected request, not an uncertain transport outcome.
        raise
    write_json(raw,response)
    pending.unlink()  # Only this adapter's completed-call marker.
    return parsed_response(response)



def generate(args):
    output = Path(args.out)
    manifest = json.loads((output/'manifest.json').read_text())
    client = Client(manifest['model'])
    results = []
    for q in read_jsonl(output/'questions.jsonl'):
        qid = q['question_id']
        proposed = cached_call(client,output/'calls'/f'{qid}-proposal',
                               json.loads((output/'requests'/f'{qid}.json').read_text()))
        variants = proposed.get('variants',[])
        if len(variants)!=1 or {v.get('kind') for v in variants} != set(KINDS):
            raise ValueError('Expected exactly one paper-informed rewrite')
        variants = sorted(variants,key=lambda v:KINDS.index(v['kind']))
        for v in variants:
            if not all(isinstance(v.get(k),str) for k in
                       ['modified_task','equivalence_rationale','feature_hypothesis','caveat','counterexample_check']):
                raise ValueError('Invalid proposal fields')
            if (type(v.get('equivalence_claim')) is not bool or not isinstance(v.get('task_contract'),list)
                    or not v['task_contract'] or not all(isinstance(item,str) and item.strip() for item in v['task_contract'])):
                raise ValueError('Missing task contract or self-assessed equivalence')
        for v in variants:
            issues = local_issues(q['original_task'],v['modified_task'])
            if not v['equivalence_claim']:
                issues.append('Gemini did not claim task equivalence')
            v.update(local_issues=issues, valid=not issues,
                     semantic_status='Self-reported equivalence; not independently verified')
        results.append({**q,'variants':variants})
        write_jsonl(output/'proposals.jsonl',results)
        print(f"Set {q['essay_set']}: "+', '.join(f"{v['kind']} valid={v['valid']}" for v in variants),flush=True)
    page = '# Gemini question edits: review before evaluation\n\n'
    for q in results:
        page+=f"## Set {q['essay_set']}\n\n### Original task\n\n{q['original_task']}\n\n"
        for v in q['variants']:
            page+=f"### {v['kind']} — valid: {v['valid']}\n\n{v['modified_task']}\n\n"
            page+=f"Hypothesis: {v['feature_hypothesis']}\n\nEquivalence: {v['equivalence_rationale']}\n\n"
            page+='Task contract:\n'+''.join(f'- {item}\n' for item in v['task_contract'])+'\n'
            page+=f"Counterexample check: {v['counterexample_check']}\n\nSelf-assessed equivalence: {v['equivalence_claim']}\n\n"
            page+=f"Semantic status: {v['semantic_status']}\n\nLocal checks: {v['local_issues']}\n\nCaveat: {v['caveat']}\n\n"
    (output/'review.md').write_text(page)


def materialize(args):
    output = Path(args.out)
    proposals = read_jsonl(output/'proposals.jsonl')
    if sorted(q['essay_set'] for q in proposals)!=sorted(MARKERS):
        raise ValueError('Need proposals for all five questions')
    frozen = {q['question_id']:q for q in read_jsonl(output/'questions.jsonl')}
    records = []
    for q in proposals:
        source = frozen.get(q['question_id'])
        if source is None or any(q.get(k)!=value for k,value in source.items()):
            raise ValueError('Frozen question/context was changed after preparation')
        raw_proposals = parsed_response(json.loads(
            (output/'calls'/f"{q['question_id']}-proposal.response.json").read_text()))
        original_variants = {v['kind']:v for v in raw_proposals['variants']}
        if {v['kind'] for v in q['variants']} != set(KINDS) or len(q['variants'])!=1:
            raise ValueError('Missing experimental conditions')
        tasks = [('original',q['original_task'])]
        for v in q['variants']:
            if v['modified_task'] != original_variants[v['kind']]['modified_task']:
                raise ValueError('Candidate was changed after review; regenerate/review in a new output directory')
            if not v['valid'] or not v['equivalence_claim'] or local_issues(q['original_task'],v['modified_task']):
                raise ValueError('A candidate failed validity screening; inspect review.md before proceeding')
            tasks.append((v['kind'],v['modified_task']))
        for condition,task in tasks:
            variant_id = hashlib.sha256(f"{q['question_id']}:{condition}".encode()).hexdigest()[:16]
            records.append({'question_id':variant_id,'base_question_id':q['question_id'],
                            'essay_set':q['essay_set'],'condition':condition,
                            'question':q['immutable_context']+task+q['immutable_suffix'],
                            'humans':[], 'note':'No historical humans stand in for modified-prompt responses.'})
    destination = output/'evaluation-data/questions.jsonl'
    write_jsonl(destination,records)
    print(f'Wrote {len(records)} conditions to {destination}. No answers generated or detector calls made.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','generate','materialize'])
    parser.add_argument('--data',default='data/sas')
    parser.add_argument('--out',default='runs/gemini-one-shot')
    parser.add_argument('--model',default=os.environ.get('GEMINI_MODEL','gemini-3.1-pro-preview'))
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
