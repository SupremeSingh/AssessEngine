import tempfile
import unittest
from pathlib import Path

from assessengine.core import question_records, split_response
from assessengine.__main__ import manifest


class BaselineTests(unittest.TestCase):
    def test_reasoning_never_becomes_an_answer(self):
        self.assertEqual(split_response('<think>still thinking', True)['answer'], '')
        self.assertEqual(split_response('maybe the answer is X', False)['status'],
                         'missing_reasoning_boundary')

    def test_complete_and_truncated_answers(self):
        result = split_response('<think>reason</think>Answer', True)
        self.assertEqual(result, dict(reasoning='reason', answer='Answer', status='complete'))
        self.assertEqual(split_response('reason</think>Partial', False)['status'], 'truncated')
        self.assertEqual(split_response('reason</think>', True)['status'], 'empty_answer')

    def test_duplicate_questions_grouped_without_editing_humans(self):
        rows = [{'question': 'What is X?', 'student_answer': '  raw answer  '},
                {'question': 'What  is X?', 'student_answer': 'another'}]
        records = question_records(rows, 1, 42)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['humans'], rows)
        self.assertEqual(records, question_records(rows, 1, 42))
        with self.assertRaises(ValueError):
            question_records(rows, 2, 42)

    def test_resume_refuses_changed_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            manifest(path, {'seed': 1})
            manifest(path, {'seed': 1})
            with self.assertRaises(ValueError):
                manifest(path, {'seed': 2})


if __name__ == '__main__':
    unittest.main()
