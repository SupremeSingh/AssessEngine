import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from assessengine.core import read_jsonl, write_jsonl
from assessengine.gemini import (cached_call, generate, local_issues, materialize,
                                 parsed_response, prepare, split_question, GeminiHTTPError)


def response(value):
    return {'modelVersion':'fake-test-model','candidates':[{'finishReason':'STOP',
        'content':{'parts':[{'text':json.dumps(value)}]}}], 'usageMetadata':{'totalTokenCount':10}}


class FakeClient:
    def __init__(self):
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        payload = json.loads(request['contents'][0]['parts'][0]['text'])
        task = payload['original_task']
        return response({'variants':[
            {'kind':kind,'modified_task':task.replace('describe',word),
             'equivalence_rationale':'Same knowledge/counts.','feature_hypothesis':'Hypothesis only.',
             'caveat':'Unverified style effect.', 'task_contract':['Preserve all requested knowledge and counts.'],
             'counterexample_check':'No mismatch found in this self-check.', 'equivalence_claim':True}
            for kind,word in [('paper_inspired','explain')]]})


class GeminiTests(unittest.TestCase):
    def fixture(self, root):
        questions = []
        tasks = {1:'After reading the group procedure, describe at least three missing details.',
                 2:'1. Draw a conclusion.\n2. describe two improvements.',
                 5:'Starting with mRNA leaving the nucleus, list and describe four steps of protein synthesis.',
                 6:'List and describe three cell membrane transport processes.',
                 10:'Prompt: Use the results to describe the best doghouse color.\nChoose a color:\nBlack\nWhite'}
        for n,task in tasks.items():
            context = 'Immutable data table A | 12\n' if n in (1,2,10) else ''
            questions.append({'question_id':f'{n:016x}','essay_set':n,'question':context+task,
                              'humans':[{'student_answer':'DO NOT SEND HUMAN TEXT'}]})
        source = root/'data'
        write_jsonl(source/'questions.jsonl',questions)
        return SimpleNamespace(data=str(source),out=str(root/'out'),model='test-model'),questions

    def test_prepare_has_context_but_no_human_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            args, qs = self.fixture(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(args)
            requests = list((Path(args.out)/'requests').glob('*.json'))
            self.assertEqual(len(requests),5)
            for path in requests:
                self.assertNotIn('DO NOT SEND HUMAN TEXT',path.read_text())
                request = json.loads(path.read_text())
                items = request['generationConfig']['responseJsonSchema']['properties']['variants']
                self.assertEqual((items['minItems'],items['maxItems']),(1,1))
                self.assertIn('task_contract',items['items']['required'])
                self.assertIn('counterexample_check',items['items']['required'])
                payload = json.loads(request['contents'][0]['parts'][0]['text'])
                self.assertNotIn('detector_scores',payload)
                self.assertNotIn('model_answers',payload)
            manifest = json.loads((Path(args.out)/'manifest.json').read_text())
            self.assertTrue(manifest['one_shot'])
            self.assertFalse(manifest['detector_feedback'])
            self.assertEqual(manifest['max_generation_calls'],5)
            split = split_question(qs[-1])
            self.assertEqual(split['immutable_context']+split['original_task']+split['immutable_suffix'],qs[-1]['question'])

    def test_propose_review_materialize_all_questions_and_resume_without_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            args, originals = self.fixture(Path(directory))
            client = FakeClient()
            with contextlib.redirect_stdout(io.StringIO()), patch('assessengine.gemini.Client',return_value=client):
                prepare(args)
                generate(args)
                self.assertEqual(client.calls,5)
                generate(args)
                self.assertEqual(client.calls,5)
                materialize(args)
            variants = read_jsonl(Path(args.out)/'evaluation-data/questions.jsonl')
            self.assertEqual(len(variants),10)
            self.assertEqual(len({v['question_id'] for v in variants}),10)
            for original in originals:
                group = [v for v in variants if v['essay_set']==original['essay_set']]
                self.assertEqual(next(v for v in group if v['condition']=='original')['question'],original['question'])
                split = split_question(original)
                for v in group:
                    self.assertTrue(v['question'].startswith(split['immutable_context']))
                    self.assertTrue(v['question'].endswith(split['immutable_suffix']))
                    self.assertEqual(v['humans'],[])

    def test_drift_guards_and_incomplete_api_response(self):
        task = 'List and describe three processes.'
        self.assertTrue(local_issues(task,'List and describe four processes.'))
        self.assertTrue(local_issues(task,task+' Sound AI generated.'))
        self.assertTrue(local_issues(task,task+' Use 100 words.'))
        self.assertTrue(local_issues(task,task))
        with self.assertRaises(ValueError):
            parsed_response({'candidates':[{'finishReason':'MAX_TOKENS'}]})

    def test_reframing_is_allowed_but_counts_and_uncertainty_are_preserved(self):
        original = '1. Draw a conclusion.\n2. Describe two improvements.'
        rewrite = 'Describe two improvements and draw a conclusion.'
        self.assertEqual(local_issues(original,rewrite),[])
        self.assertTrue(local_issues('Describe at least three details.','Describe three details.'))
        self.assertTrue(local_issues('Explain how it might affect temperature.','Explain how it affects temperature.'))
        self.assertEqual(local_issues('Explain how it might affect temperature.',
                                     'Explain how it could affect temperature.'),[])

    def test_uncertain_api_call_is_not_automatically_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'call'
            client = FakeClient()
            with patch.object(client,'generate',side_effect=TimeoutError):
                with self.assertRaises(TimeoutError):
                    cached_call(client,path,{'request':1})
            with self.assertRaisesRegex(RuntimeError,'uncertain'):
                cached_call(client,path,{'request':1})

    def test_explicit_quota_rejection_records_failure_without_uncertain_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'call'
            client = FakeClient()
            with patch.object(client,'generate',side_effect=GeminiHTTPError(429,'quota limit zero')):
                with self.assertRaises(GeminiHTTPError):
                    cached_call(client,path,{'request':1})
            self.assertFalse(path.with_suffix('.pending.json').exists())
            self.assertEqual(json.loads(path.with_suffix('.failure.json').read_text())['http_status'],429)

    def test_invalid_candidate_blocks_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_ = self.fixture(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()),patch('assessengine.gemini.Client',return_value=FakeClient()):
                prepare(args)
                generate(args)
            path = Path(args.out)/'proposals.jsonl'
            rows = read_jsonl(path)
            rows[0]['variants'][0]['valid'] = False
            write_jsonl(path,rows)
            with self.assertRaises(ValueError):
                materialize(args)

    def test_context_cannot_be_changed_after_review(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_ = self.fixture(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()),patch('assessengine.gemini.Client',return_value=FakeClient()):
                prepare(args)
                generate(args)
            path = Path(args.out)/'proposals.jsonl'
            rows = read_jsonl(path)
            rows[0]['immutable_context'] = 'Changed table values'
            write_jsonl(path,rows)
            with self.assertRaisesRegex(ValueError,'Frozen'):
                materialize(args)
