import unittest

from assessengine.diagnose import summarize


class DiagnosticTests(unittest.TestCase):
    def test_threshold_grouping_and_counts(self):
        rows = [
            {'id': 'E01.Q01.A00', 'question': 'Explain X', 'student_answer': 'word ' * 49},
            {'id': 'E01.Q01.A01', 'question': 'Explain  X', 'student_answer': 'word ' * 50},
            {'id': 'E02.Q01.A00', 'question': 'Explain X', 'student_answer': 'word ' * 100},
            {'id': 'E02.Q02.A00', 'question': 'Explain Y', 'student_answer': ''},
        ]
        summary, questions = summarize(rows)
        self.assertEqual(summary['eligible_answers'], 2)
        self.assertEqual(summary['distinct_question_wordings'], 2)
        self.assertEqual(summary['questions_with_eligible_answers']['1'], 1)
        self.assertEqual(sum(summary['answer_length_counts'].values()), 4)
        self.assertEqual(questions[0]['source_question_ids'], 'E01.Q01,E02.Q01')
        self.assertEqual(questions[0]['median_words'], 50)

    def test_empty_dataset(self):
        summary, questions = summarize([])
        self.assertEqual(summary['eligible_answers'], 0)
        self.assertIsNone(summary['median_words'])
        self.assertEqual(questions, [])
