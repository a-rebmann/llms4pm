
from collections import defaultdict
import pandas as pd
from const import DATA_ROOT
from pm4py.objects.log.obj import EventLog, Trace, Event
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
from Declare4Py.D4PyEventLog import D4PyEventLog



def get_declare_constraints(string_traces: list[str]) -> pd.DataFrame:
    pm4pylog = EventLog()
    for i, trace in enumerate(string_traces):
        pm4pytrace = Trace(attributes={"concept:name": f"trace_{i}"})
        for j, event in enumerate(trace):
            pm4pyevent = Event()
            pm4pyevent["concept:name"] = event
            pm4pyevent["time:timestamp"] = j
            pm4pytrace.append(pm4pyevent)
        pm4pylog.append(pm4pytrace)
    pm4pylog._properties['pm4py:param:timestamp_key'] = 'time:timestamp'
    pm4pylog._properties['pm4py:param:activity_key'] = 'concept:name'
    event_log = D4PyEventLog(log=pm4pylog)
    discovery = DeclareMiner(log=event_log, consider_vacuity=False, min_support=0.00001, itemsets_support=0.00001, max_declare_cardinality=3)
    discovered_model: DeclareModel = discovery.run()
    constraint_strs: list[str] = discovered_model.serialized_constraints
    constraint_strs = [c.replace("|", "").strip() for c in constraint_strs if "Absence" not in c and "Existence" not in c]
    return constraint_strs


if __name__ == "__main__":
    model_df = pd.read_csv(DATA_ROOT / "process_behavior_corpus.csv")
    model_df["string_traces"] = model_df["string_traces"].apply(eval)
    # drop pn column
    model_df = model_df.drop(columns=["pn", "string_traces", "name", "language"])
    #model_df["declare"] = model_df["string_traces"].apply(lambda x: get_declare_constraints(x))
    model_df.to_csv(DATA_ROOT / "S-PMD.csv", index=False)