from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).parents[0].resolve()
DATA_ROOT = PROJECT_ROOT #/ "data"
DATA_DATASET = DATA_ROOT / "models"
DATA_RESULTS = DATA_ROOT / "results"
EVAL_PATH = DATA_ROOT  / "eval" # changed for simplicity 
BPMN2_NAMESPACE = "http://b3mn.org/stencilset/bpmn2.0#"


anomaly_types = {
    0: 'out_of_order',
    1: 'superfluous_activity',
    2: 'missing_activity',
}


GPT_MODEL = "gpt-3.5-turbo"
LLAMA_2_MODEL = "meta-llama/Llama-2-7b-chat-hf"

LLAMA_3_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA_3_SMALL = "meta-llama/Llama-3.2-1B-Instruct"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
GEMMA_MODEL = "google/gemma-7b-it"


OUT_OF_ORDER = "out_of_order"
TRACE_ANOMALY = "trace_anomaly"
NEXT_ACTIVITY = "next_activity"
DFG_GENERATION = "dfg_generation"
PT_GENERATION = "pt_generation"

TASKS = [OUT_OF_ORDER, TRACE_ANOMALY, NEXT_ACTIVITY, DFG_GENERATION, PT_GENERATION]