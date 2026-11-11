import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from assessengine.sas import SETS, clean_prompt, prepare_records, response_prompt
from assessengine.core import read_jsonl, write_jsonl
from assessengine.snapjudge import export


class SASTests(unittest.TestCase):
    def rows(self):
        return [{'essay_set': n, 'Id': f'{n}-{i}', '_split': 'train',
                 'essay_text': ('word ' * length) + str(i),
                 'prompt': f'Question {n}' + ('\nRubric for X\nKey Elements:\nSECRET' if n in (5, 6) else ''),
                 'prompt_&_source_essay_urls': 'https://example.org/prompt'}
                for n in SETS for i, length in enumerate((48, 49, 55, 60))]

    def test_no_rubric_leakage_and_exact_raw_answers(self):
        rows = self.rows()
        questions, coverage = prepare_records(rows, 2)
        self.assertEqual([q['essay_set'] for q in questions], list(SETS))
        for q in questions:
            self.assertNotIn('SECRET', response_prompt(q))
            self.assertNotIn('Rubric', q['question'])
            self.assertEqual(len(q['humans']), 2)
            for h in q['humans']:
                self.assertIn(h['student_answer'], [r['essay_text'] for r in rows])
                self.assertGreaterEqual(h['words'], 50)
        self.assertTrue(all(c['eligible_answers'] == 3 for c in coverage))

    def test_deduplication_and_order_independent_sampling(self):
        rows = self.rows()
        a, coverage = prepare_records(rows + [rows[2]], 2)
        b, _ = prepare_records(list(reversed(rows)), 2)
        self.assertEqual(a, b)
        self.assertEqual(coverage[0]['unique_answers'], 4)
        with self.assertRaises(ValueError):
            prepare_records(rows, 4)

    def test_missing_or_unexpected_rubric_fails(self):
        with self.assertRaises(ValueError):
            clean_prompt('Question only', 5)
        with self.assertRaises(ValueError):
            clean_prompt('Question\nRubric for X', 1)

    def test_context_and_tables_preserved(self):
        text = 'Procedure:\n1. Measure.\nTable: A | 10\nExplain.'
        self.assertEqual(clean_prompt(text, 1), text)
        self.assertIn(text, response_prompt({'question': text}))
        with self.assertRaises(ValueError):
            response_prompt({'question': text}, 20, 40)

    def test_prepared_sas_is_compatible_with_snapjudge_export(self):
        questions, _ = prepare_records(self.rows(), 2)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_jsonl(run / 'questions.jsonl', questions)
            write_jsonl(run / 'answers.jsonl', [
                {'id': 'model-1', 'question_id': questions[0]['question_id'],
                 'answer': 'answer ' * 60, 'status': 'complete'}])
            with contextlib.redirect_stdout(io.StringIO()):
                output = export(run)
            index = read_jsonl(output / 'index.jsonl')
            self.assertEqual(len(index), 11)
            self.assertTrue(all(r['excluded_reason'] is None for r in index))
            self.assertEqual(len(list((output / 'documents').glob('*.txt'))), 11)
