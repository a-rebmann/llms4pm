import numpy as np
import pandas as pd


def _get_weak_order_matrix(log, as_df=False):
    unique_event_classes = list(set([e for c in log for e in c]))
    wom = np.zeros(shape=(len(unique_event_classes), len(unique_event_classes)))
    for case in log:
        activities = [e for e in case]
        for i in range(0, len(activities) - 1):
            for j in range(i + 1, len(activities)):
                wom[unique_event_classes.index(activities[i]), unique_event_classes.index(
                    activities[j])] += 1
    if as_df:
        return pd.DataFrame(wom, columns=unique_event_classes, index=unique_event_classes)
    return wom


def get_behavioral_profile_as_df(log, as_df=False):
    wom = _get_weak_order_matrix(log, as_df=True)
    cols = wom.columns
    wom = wom.values
    wom_len = len(wom)
    res = np.empty((wom_len, wom_len), dtype=float)
    strict_order, reverse_strict_order, exclusive, interleaving = set(), set(), set(), set()
    for i in range(wom_len):
        for j in range(wom_len):
            if wom[i, j] == 0 and wom[j, i] != 0:
                res[j, i] = 0
                reverse_strict_order.add((cols[i], cols[j]))
                continue
            if wom[j, i] == 0 and wom[i, j] != 0:
                res[j, i] = 1
                strict_order.add((cols[i], cols[j]))
                continue
            if wom[i, j] != 0 and wom[j, i] != 0:
                res[j, i] = 2
                interleaving.add((cols[i], cols[j]))
                continue
            if wom[i, j] == 0 and wom[j, i] == 0:
                res[j, i] = 3
                exclusive.add((cols[i], cols[j]))
                continue
    if as_df:
        return pd.DataFrame(res, columns=cols, index=cols).replace([0, 1, 2, 3, 4], ['->', "<-", "||", "+", ""])
    else:
        return strict_order, reverse_strict_order, exclusive, interleaving
