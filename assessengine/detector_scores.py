"""Continuous SnapJudge decision scores: raw margins, never calibrated probabilities."""
import math
from statistics import mean


def continuous_scores(result):
    if result.get('score_type') != 'logit3':
        return None
    windows = result.get('window_scores',[])
    if not windows or not all(k in result for k in ['thr_nonhuman','thr_generated']):
        return None
    if not all('nonhuman' in window and 'gen_minus_pol' in window for window in windows):
        return None
    thresholds = [float(result[k]) for k in ['thr_nonhuman','thr_generated']]
    values = [(float(w['nonhuman']),float(w['gen_minus_pol'])) for w in windows]
    if not all(math.isfinite(x) for x in thresholds+[n for pair in values for n in pair]):
        raise ValueError('Non-finite detector scores')
    margins = [min(nonhuman-thresholds[0], gen_minus_pol-thresholds[1])
               for nonhuman,gen_minus_pol in values]
    return {'score_type':'logit3', 'windows':len(windows),
            'thr_nonhuman':thresholds[0], 'thr_generated':thresholds[1],
            'mean_nonhuman_score':mean(v[0] for v in values),
            'mean_generated_vs_polished_score':mean(v[1] for v in values),
            'mean_generated_decision_margin':mean(margins),
            'interpretation':'Derived signed threshold margin, not a probability. '
                             'For a single window, positive means both generated-class thresholds are exceeded.'}
