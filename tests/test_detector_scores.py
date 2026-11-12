import math
import unittest

from assessengine.detector_scores import continuous_scores


class DetectorScoreTests(unittest.TestCase):
    def test_signed_margin_uses_both_generated_thresholds(self):
        base = {'score_type':'logit3','thr_nonhuman':12,'thr_generated':10,
                'window_scores':[{'nonhuman':16,'gen_minus_pol':13}]}
        self.assertEqual(continuous_scores(base)['mean_generated_decision_margin'],3)
        base['window_scores'][0]['gen_minus_pol'] = 8
        self.assertEqual(continuous_scores(base)['mean_generated_decision_margin'],-2)

    def test_unknown_score_is_not_invented_and_nan_rejected(self):
        self.assertIsNone(continuous_scores({'score_type':'unknown'}))
        self.assertIsNone(continuous_scores({'score_type':'logit3','thr_nonhuman':12,'thr_generated':10,
                                            'window_scores':[{'class':'generated'}]}))
        with self.assertRaises(ValueError):
            continuous_scores({'score_type':'logit3','thr_nonhuman':12,'thr_generated':10,
                               'window_scores':[{'nonhuman':math.nan,'gen_minus_pol':13}]})
