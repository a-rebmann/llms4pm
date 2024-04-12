import gc
import json
import logging
import signal
import time

import pm4py
from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.algo.simulation.playout.petri_net.algorithm import Variants
from pm4py.objects.log.obj import EventLog, Trace
from func_timeout import func_timeout, FunctionTimedOut

from sadsnap.bpmnjsonanalyzer import fromJSON
from sadsnap.jsontopetrinetconverter import JsonToPetriNetConverter

_logger = logging.getLogger(__name__)


def create_variant_log(log):
    variant_log = EventLog()
    seen = set()
    already_counted_loop = False
    for trace in log:
        trace_labels = tuple([x["concept:name"] for x in trace if x["concept:name"] != ""])
        if trace_labels not in seen:
            trace_cpy = Trace()
            for event in trace:
                if event["concept:name"] != "":
                    trace_cpy.append(event)
            variant_log.append(trace_cpy)
            seen.add(trace_labels)
            if 0 < len(trace_labels) == len(set(trace_labels)) and not already_counted_loop:
                already_counted_loop = True
    return variant_log


def create_log(net, im, fm, model_elements, loops=False):
        log = pm4py.play_out(net, im, fm, variant=Variants.BASIC_PLAYOUT)
        for trace in log:
            for event in trace:
                e_id = event["concept:name"]
                event["eid"] = e_id
                event["concept:name"] = model_elements.loc[e_id,"label"]
                event["category"] = model_elements.loc[e_id, "category"]
        variant_log = create_variant_log(log)
        if not loops:
            played_out_log = create_log_without_loops(variant_log)
        else:
            played_out_log = variant_log
        return played_out_log


def create_log_without_loops(log):
    log_no_loops = EventLog()
    for trace in log:
        trace_labels = [x["concept:name"] for x in trace]
        if 0 < len(trace_labels) == len(set(trace_labels)):
            log_no_loops.append(trace)
    return log_no_loops


def _get_json_from_row(row_tuple):
    return json.loads(row_tuple.model_json)


class Model2LogConverter:

    def __init__(self):
        self.converter = JsonToPetriNetConverter()
        self.done = 0
        self.loop_counter = 0

    def alarm_handler(self, signum, frame):
        raise Exception("timeout")

    def check_soundness(self, df):
        df["sound"] = df.apply(lambda x: self.soundness_check(x), axis=1)
        return df

    def create_log(self, df):
        df["traces"] = df.apply(lambda x: self.log_creation_check(x), axis=1)
        return df

    def soundness_check(self, row):
        if row.pn:
            _logger.info("Soundness check. " + str(row.model_id) + "; Number " + str(self.done))
            net, im, fm = row.pn
            start = time.time()
            try:
                res = func_timeout(10, woflan.apply,
                                   args=(net, im, fm, {woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
                                                       woflan.Parameters.PRINT_DIAGNOSTICS: False,
                                                       woflan.Parameters.RETURN_DIAGNOSTICS: False}))
                _logger.info("Result: " + str(row.model_id) + " sound? " + str(res))
            except FunctionTimedOut as ex:
                _logger.warning("Time out during soundness checking.", ex)
                res = False
            except Exception as ex:
                _logger.warning("Error during soundness checking.", ex)
                res = False
            finally:
                stop = time.time()
                completed_in = round(stop - start, 2)
                if completed_in > 20:
                    _logger.error("Timeout not working!! " + str(row.model_id))
            self.done += 1
            if self.done % 5000 == 0:
                _logger.info("Collect garbage...")
                gc.collect()
                _logger.info("GC done.")
            return res
        else:
            return False

    def convert_to_pn_lamda(self, row):
        json_str = _get_json_from_row(row)
        try:
            f, l, _ = fromJSON(json_str)
            follows_ = {}
            labels_ = {}
            for e in l:
                labels_[str(row.model_id) + str(e)] = l[e]
            for e in f:
                follows_[str(row.model_id) + str(e)] = [str(row.model_id) + str(e) for e in f[e]]
            return self.converter.convert_from_parsed(follows_, labels_)
        except KeyError as ke:
            _logger.info("Error during conversion from bpmn to Petri net." + str(ke))
            return None
        except Exception as ex:
            _logger.info("Error during conversion from bpmn to Petri net." + str(ex))
            return None

    def convert_models_to_pn_df(self, df_bpmn):
        df_bpmn["pn"] = df_bpmn.apply(lambda x: self.convert_to_pn_lamda(x), axis=1)
        return df_bpmn

    def create_variant_log(self, log):
        variant_log = EventLog()
        seen = set()
        already_counted_loop = False
        for trace in log:
            trace_labels = tuple([x["concept:name"] for x in trace if x["concept:name"] != ""])
            if trace_labels not in seen:
                trace_cpy = Trace()
                for event in trace:
                    if event["concept:name"] != "":
                        trace_cpy.append(event)
                variant_log.append(trace_cpy)
                seen.add(trace_labels)
                if 0 < len(trace_labels) == len(set(trace_labels)) and not already_counted_loop:
                    self.loop_counter += 1
                    already_counted_loop = True
        return variant_log

    def log_creation_check(self, row):
        played_out_log = None
        if row.sound:
            start = time.time()
            signal.signal(signal.SIGALRM, self.alarm_handler)
            signal.alarm(10)
            try:
                net, im, fm = row.pn
                log = pm4py.play_out(net, im, fm, variant=Variants.EXTENSIVE)
                variant_log = self.create_variant_log(log)
                played_out_log = variant_log
            except Exception as ex:
                _logger.warning(str(ex))
            finally:
                signal.alarm(0)
                stop = time.time()
                completed_in = round(stop - start, 2)
                if completed_in > 15:
                    _logger.error("Timeout not working!!")
        return played_out_log
