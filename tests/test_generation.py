import unittest

from assessengine.generation import validate_resume_source

from assessengine.generation import attempt_seed, next_attempt, qualifies, quota_report


def row(sample=0, attempt=1, words=60, status='complete'):
    return dict(question_id='q', sample=sample, attempt=attempt,
                answer='word ' * words, status=status)


class GenerationTests(unittest.TestCase):
    def test_variable_quotas_and_safe_import(self):
        questions = [{'question_id': str(n)} for n in [1,2,5,6,10]]
        targets = {q['question_id']: 21 if q['question_id']=='10' else 50 for q in questions}
        previous = {'samples':20, 'revision':'fixed', 'seed':42}
        current = {**previous, 'samples':50, 'samples_by_question': targets}
        validate_resume_source(previous, current, questions, [])
        summary = quota_report(questions, [], targets, 5)
        self.assertEqual(sum(q['missing'] for q in summary['questions']), 221)
        self.assertFalse(summary['all_quotas_met'])
        with self.assertRaisesRegex(ValueError, 'different generation settings'):
            validate_resume_source(previous, {**current, 'seed':43}, questions, [])
        with self.assertRaisesRegex(ValueError, 'must not shrink'):
            validate_resume_source(previous, {**current, 'samples_by_question': {'10':19}}, questions, [])

    def test_only_completed_long_final_answers_qualify(self):
        self.assertTrue(qualifies(row(words=50)))
        self.assertFalse(qualifies(row(words=49)))
        self.assertFalse(qualifies(row(status='truncated')))

    def test_resume_retries_then_moves_to_next_slot(self):
        rows = [row(words=49)]
        self.assertEqual(next_attempt(rows, 2, 3), (0, 2))
        rows.append(row(attempt=2))
        self.assertEqual(next_attempt(rows, 2, 3), (1, 1))
        rows.append(row(sample=1))
        self.assertIsNone(next_attempt(rows, 2, 3))

    def test_retry_cap_and_unmet_quota(self):
        rows = [row(words=49, attempt=n) for n in (1, 2, 3)]
        self.assertIsNone(next_attempt(rows, 1, 3))
        summary = quota_report([{'question_id': 'q'}], rows, 1, 3)
        self.assertFalse(summary['all_quotas_met'])
        self.assertEqual(summary['questions'][0]['rejected'], 3)
        self.assertEqual(next_attempt(rows, 2, 3), (1, 1))

    def test_seeds_and_duplicate_protection(self):
        self.assertEqual(attempt_seed(42, 'q', 1, 2), attempt_seed(42, 'q', 1, 2))
        self.assertNotEqual(attempt_seed(42, 'q', 1, 2), attempt_seed(42, 'q', 1, 3))
        with self.assertRaises(ValueError):
            next_attempt([row(), row()], 1, 3)

    def test_fifty_slots_with_retry_remain_fifty_accepted(self):
        rows = []
        while (task := next_attempt(rows, 50, 5)) is not None:
            sample, attempt = task
            # Simulate first-attempt failures on every second slot.
            words = 49 if sample % 2 == 0 and attempt == 1 else 60
            rows.append(row(sample=sample, attempt=attempt, words=words))
        summary = quota_report([{'question_id': 'q'}], rows, 50, 5)
        self.assertTrue(summary['all_quotas_met'])
        self.assertEqual(summary['questions'][0]['accepted'], 50)
        self.assertEqual(summary['questions'][0]['attempts'], 75)
        self.assertEqual(summary['questions'][0]['rejected'], 25)
