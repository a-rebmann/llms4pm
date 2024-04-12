def is_missing(affected, trace, strict_order, reverse_strict_order, interleaving):
    res = False
    for trace_activity in trace:
        if (trace_activity, affected[0]) in strict_order\
                or (trace_activity, affected[0]) in reverse_strict_order\
                or (trace_activity, affected[0]) in interleaving:
            res = True
    return res


def is_superfluous(affected, trace, true_activity_relations):
    res = False
    for trace_activity in trace:
        if (trace_activity, affected[0]) in true_activity_relations:
            res = True
    return res


def is_out_of_order(affected, true_activity_relations):
    res = len(affected) > 1 and (affected[1], affected[0]) in true_activity_relations
    return res


def is_true_anomaly(anomaly_type, affected, trace, strict_order, reverse_strict_order, exclusive, interleaving):
    if anomaly_type == 0:
        # if event2 never follows event1 in the original model, it is a true order anomaly
        return is_out_of_order(affected, strict_order)
    if anomaly_type == 1:
        # if event1 excludes any each other event in the original model, it is a true superfluous anomaly
        return is_superfluous(affected, trace, exclusive)
    if anomaly_type == 2:
        # if event1 interleaves with any event in the trace or is in strict order relation in the original model, it is a true missing anomaly
        return is_missing(affected, trace, strict_order, reverse_strict_order, interleaving)
    return False


def has_cooccurrence_relation(affected, log):
    # checks if every trace that has event1 also has event2
    for variant in log:
        has_event1 = False
        has_event2 = False
        for event in variant:
            if event == affected[0]:
                has_event1 = True
            if event == affected[1]:
                has_event2 = True
        if has_event1 and not has_event2:
            return False
    return True


def has_exclusion_relation(affected, log):
    # checks if there is a trace where event1 and event2 both occur
    for variant in log:
        has_event1 = False
        has_event2 = False
        for event in variant:
            if event == affected[0]:
                has_event1 = True
            if event == affected[1]:
                has_event2 = True
        if has_event1 and has_event2:
            return False
    return True


def has_follows_relation(affected, log):
    # checks if there is a trace where event1 occurs before event2 in the log
    for variant in log:
        for i in range(len(variant) - 1):
            if variant[i] == affected[0]:
                for j in range(i + 1, len(variant)):
                    if variant[j] == affected[2]:
                        return True
    return False
