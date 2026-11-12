from decimal import Decimal
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from assessengine.pangram import PangramHTTPError, credit_estimate, inputs, main, make_plan, process
from assessengine.core import write_json, write_jsonl


def row(key='a', words=80):
    return {'cache_key': key, 'text': 'word ' * words, 'credits_estimate': credit_estimate('word ' * words),
            'authorship': 'human'}


class FakeClient:
    def __init__(self, balance='150', fail_post=False, stage='STAGE_SUCCESS'):
        self.available = Decimal(balance)
        self.fail_post = fail_post
        self.stage = stage
        self.posts = 0

    def balance(self):
        return self.available

    def request(self, method, path, **kwargs):
        if method == 'POST':
            self.posts += 1
            if self.fail_post:
                raise TimeoutError('response lost')
            assert kwargs['json']['public_dashboard_link'] is False
            assert kwargs['json']['model'] == 'pangram-4'
            return {'task_id': 'task-1'}
        return {'stage': self.stage}


class PangramTests(unittest.TestCase):
    def test_partial_selection_balances_within_question_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'generation_coverage.json', {'all_quotas_met': False})
            write_jsonl(root / 'questions.jsonl', [
                {'question_id': q, 'humans': [
                    {'id': f'{q}-h{i}', 'student_answer': 'human ' * 60} for i in range(3)]}
                for q in ('q1', 'q2', 'q3')])
            saved = [dict(id=f'{q}-a{i}', question_id=q, sample=i, attempt=1,
                          status='complete', answer='answer ' * 60)
                     for q, count in [('q1', 3), ('q2', 1)] for i in range(count)]
            saved.append(dict(id='short', question_id='q2', sample=1, status='complete', answer='short'))
            write_jsonl(root / 'answers.jsonl', saved)
            before = (root / 'answers.jsonl').read_bytes()
            with self.assertRaises(ValueError):
                inputs(root)
            selected = inputs(root, allow_partial=True)
            plan = make_plan(selected, {})
            self.assertEqual(plan['by_authorship'], {'human': 4, 'ai': 4})
            self.assertEqual(plan['by_question']['q2'], {'human': 1, 'ai': 1})
            self.assertNotIn('q3', plan['by_question'])
            self.assertEqual([r['id'] for r in selected if r['question_id'] == 'q2'], ['q2-h0', 'q2-a0'])
            self.assertEqual(inputs(root, allow_partial=True), selected)
            self.assertEqual((root / 'answers.jsonl').read_bytes(), before)

    def test_credit_rounding_and_deduplication(self):
        self.assertEqual(credit_estimate('word ' * 100), 1)
        self.assertEqual(credit_estimate('word ' * 101), 2)
        plan = make_plan([row(), row()], {})
        self.assertEqual(plan['documents'], 2)
        self.assertEqual(plan['unique_texts'], 1)
        self.assertEqual(plan['estimated_new_credits'], 1)

    def test_ledger_and_run_budget(self):
        self.assertFalse(make_plan([row()], {}, max_credits=0)['within_budget'])
        ledger = {'old': {'credits_estimate': 1500}}
        self.assertFalse(make_plan([row()], ledger)['within_budget'])
        ledger['a'] = {'credits_estimate': 1}
        self.assertEqual(make_plan([row()], ledger)['estimated_new_credits'], 0)

    def test_reservation_precedes_post_and_reuses_success(self):
        ledger, saved = {}, []
        client = FakeClient()
        process(row(), ledger, lambda: saved.append(ledger['a']['state']), client)
        self.assertEqual(saved, ['submitting', 'polling', 'done'])
        process(row(), ledger, lambda: None, client)
        self.assertEqual(client.posts, 1)

    def test_lost_post_response_never_reposts(self):
        ledger = {}
        client = FakeClient(fail_post=True)
        with self.assertRaises(TimeoutError):
            process(row(), ledger, lambda: None, client)
        with self.assertRaisesRegex(RuntimeError, 'Uncertain previous submission'):
            process(row(), ledger, lambda: None, client)
        self.assertEqual(client.posts, 1)

    def test_saved_task_resumes_polling_without_post(self):
        ledger = {'a': {'state': 'polling', 'task_id': 'saved', 'credits_estimate': 1}}
        client = FakeClient()
        process(row(), ledger, lambda: None, client)
        self.assertEqual(client.posts, 0)
        self.assertEqual(ledger['a']['state'], 'done')

    def test_live_balance_blocks_paid_call(self):
        client = FakeClient(balance='75.00')
        ledger = {}
        with self.assertRaisesRegex(RuntimeError, 'balance is too low'):
            process(row(), ledger, lambda: None, client)
        self.assertEqual(client.posts, 0)
        self.assertEqual(ledger, {})

    def test_failed_task_is_not_paid_for_again(self):
        client = FakeClient(stage='STAGE_FAILED')
        ledger = {}
        process(row(), ledger, lambda: None, client)
        process(row(), ledger, lambda: None, client)
        self.assertEqual(ledger['a']['state'], 'failed')
        self.assertEqual(client.posts, 1)

    def test_missing_balance_requires_opt_in(self):
        client = FakeClient()
        with patch.object(client, 'balance', side_effect=PangramHTTPError(404, 'GET', '/balance')):
            with self.assertRaises(PangramHTTPError):
                process(row(), {}, lambda: None, client)
        self.assertEqual(client.posts, 0)

    def test_missing_balance_opt_in_only_accepts_balance_404(self):
        client, ledger = FakeClient(), {}
        with patch.object(client, 'balance', side_effect=PangramHTTPError(404, 'GET', '/balance')) as balance, \
             contextlib.redirect_stdout(io.StringIO()):
            process(row(), ledger, lambda: None, client, allow_missing_balance=True)
            process(row(key='b'), ledger, lambda: None, client, allow_missing_balance=True)
        self.assertEqual(balance.call_count, 1)
        self.assertEqual(client.posts, 2)
        self.assertEqual(ledger['a']['balance_guard'], 'unavailable_404')
        for status, path in [(401, '/balance'), (403, '/balance'), (429, '/balance'),
                             (500, '/balance'), (404, '/task')]:
            blocked = FakeClient()
            with patch.object(blocked, 'balance', side_effect=PangramHTTPError(status, 'GET', path)):
                with self.assertRaises(PangramHTTPError):
                    process(row(), {}, lambda: None, blocked, allow_missing_balance=True)
            self.assertEqual(blocked.posts, 0)

    def test_opt_in_still_enforces_live_balance_when_available(self):
        client = FakeClient(balance='75.00')
        with self.assertRaisesRegex(RuntimeError, 'balance is too low'):
            process(row(), {}, lambda: None, client, allow_missing_balance=True)
        self.assertEqual(client.posts, 0)

    def test_plan_never_constructs_api_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ['pangram', 'plan', '--run', str(root), '--ledger', str(root / 'ledger.json')]
            with patch('sys.argv', args), patch('assessengine.pangram.inputs', return_value=[row()]), \
                 patch('assessengine.pangram.Client') as client, contextlib.redirect_stdout(io.StringIO()):
                main()
            client.assert_not_called()
            plan = json.loads((root / 'pangram/plan.json').read_text())
            self.assertTrue(plan['within_budget'])
            self.assertEqual(plan['reserve_credits'], 1500)
