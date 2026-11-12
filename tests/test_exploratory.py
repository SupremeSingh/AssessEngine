import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from assessengine.core import read_jsonl, write_jsonl
from assessengine.exploratory import prepare, generate, materialize
from assessengine.snapjudge import export


class ExploratoryTests(unittest.TestCase):
    def test_frozen_rewrites_export_ai_only_and_reject_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposals = root/'proposals.jsonl'
            rows = [{'question_id': str(n), 'essay_set': n, 'question': 'Original context',
                     'rewritten_question': f'Exact rewritten context {n}'} for n in [1,2,5,6,10]]
            write_jsonl(proposals, rows)
            args = SimpleNamespace(proposals=str(proposals), data=str(root/'data'))
            with contextlib.redirect_stdout(io.StringIO()):
                materialize(args)
                materialize(args)
            questions = read_jsonl(root/'data/questions.jsonl')
            self.assertEqual([q['question'] for q in questions],
                             [r['rewritten_question'] for r in rows])
            self.assertTrue(all(q['humans'] == [] for q in questions))
            write_jsonl(root/'run/questions.jsonl', questions)
            write_jsonl(root/'run/answers.jsonl', [
                {'id': f'{q["question_id"]}:{i}:1', 'question_id': q['question_id'],
                 'answer': 'word '*60, 'status': 'complete'} for q in questions for i in range(20)])
            with contextlib.redirect_stdout(io.StringIO()):
                output = export(root/'run')
            self.assertEqual(json.loads((output/'coverage.json').read_text()), {'ai:eligible':100})
            rows[0]['rewritten_question'] = 'Changed rewrite'
            write_jsonl(proposals, rows)
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                materialize(args)

    def test_minimal_requests_no_limits_no_humans_and_one_shot_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_jsonl(root/'data/questions.jsonl',[
                {'question_id':str(n),'essay_set':n,'question':f'Question {n} with full context',
                 'humans':[{'text':'SECRET HUMAN ANSWER'}]} for n in [1,2,5,6,10]])
            args=SimpleNamespace(data=str(root/'data'),out=str(root/'out'),model='test-model')
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(args)
            for p in (root/'out/requests').glob('*.json'):
                request=json.loads(p.read_text())
                self.assertNotIn('maxOutputTokens',request['generationConfig'])
                self.assertNotIn('SECRET HUMAN ANSWER',p.read_text())
                self.assertNotIn('Biber',p.read_text())
                self.assertNotIn('80–120',p.read_text())
            def respond(request):
                value={'rewritten_question':'A creative rewrite.', 'rationale':'Exploratory effect.',
                       'meaning_preservation_note':'Unverified suggestion.'}
                return {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(value)}]}}]}
            with patch('assessengine.exploratory.Client') as client, contextlib.redirect_stdout(io.StringIO()):
                client.return_value.generate.side_effect=respond
                generate(args)
                generate(args)
                self.assertEqual(client.return_value.generate.call_count,5)
            self.assertEqual(len(read_jsonl(root/'out/proposals.jsonl')),5)
