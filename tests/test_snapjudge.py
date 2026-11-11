import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from assessengine.core import read_jsonl, write_jsonl
from assessengine.snapjudge import export, report, run_batch


class SnapJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.long_text = 'word ' * 50
        write_jsonl(self.run / 'questions.jsonl', [
            {'question_id': 'q1', 'humans': [
                {'id': 'h1', 'student_answer': self.long_text},
                {'id': 'h2', 'student_answer': 'word ' * 49}]}])
        write_jsonl(self.run / 'answers.jsonl', [
            {'id': 'a1', 'question_id': 'q1', 'answer': self.long_text, 'status': 'complete'},
            {'id': 'a2', 'question_id': 'q1', 'answer': self.long_text, 'status': 'truncated'}])

    def exported(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return export(self.run)

    def test_length_boundary_and_incomplete_exclusions(self):
        output = self.exported()
        rows = read_jsonl(output / 'index.jsonl')
        self.assertEqual([r['excluded_reason'] for r in rows],
                         [None, 'under_50_words', None, 'incomplete_generation'])
        self.assertEqual(len(list((output / 'documents').glob('*.txt'))), 2)
        for path in (output / 'documents').glob('*.txt'):
            self.assertEqual(path.read_text(), self.long_text)
        self.assertEqual(self.exported(), output)

    def test_changed_source_refused(self):
        self.exported()
        write_jsonl(self.run / 'answers.jsonl', [])
        with self.assertRaisesRegex(ValueError, 'Inputs changed'):
            self.exported()

    def test_join_uses_filename_and_reports_missing_results(self):
        output = self.exported()
        rows = read_jsonl(output / 'index.jsonl')
        with (output / 'results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['file', 'verdict', 'ai_generated_share', 'error'])
            writer.writeheader()
            writer.writerow(dict(file=rows[2]['file'], verdict='AI-generated', ai_generated_share=1, error=''))
        with contextlib.redirect_stdout(io.StringIO()):
            report(self.run)
        result = json.loads((output / 'report.json').read_text())
        self.assertEqual(result['human']['missing_results'], 1)
        self.assertEqual(result['ai']['scored'], 1)
        self.assertEqual(result['ai']['mean_generated_window_share'], 1)

    def test_official_client_receives_key_path_and_no_save(self):
        key = self.run / 'secret.key'
        key.write_text('sjk_private_test_value')
        fake_requests = SimpleNamespace(get=lambda *a, **kw: None)
        with patch.dict('sys.modules', {'requests': fake_requests}), \
             patch('assessengine.snapjudge.subprocess.run') as call, \
             patch('assessengine.snapjudge.Path.exists', return_value=True), \
             patch('assessengine.snapjudge.digest', return_value='hash'), \
             patch('assessengine.snapjudge.export', return_value=self.run), \
             patch('assessengine.snapjudge.write_json'), \
             patch('assessengine.snapjudge.Path.glob', return_value=iter([self.run / 'sample.txt'])):
            call.return_value.returncode = 2
            # Avoid report: it depends on results not produced by this mocked process.
            with patch('assessengine.snapjudge.report'):
                run_batch(SimpleNamespace(key_file=str(key), run=str(self.run), concurrency=3))
            command = call.call_args.args[0]
            self.assertIn('--no-save', command)
            self.assertIn(str(key.resolve()), command)
            self.assertNotIn(key.read_text(), command)
