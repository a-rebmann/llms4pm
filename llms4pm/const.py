from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).parents[2].resolve()
DATA_ROOT = PROJECT_ROOT / "data"
DATA_DATASET = DATA_ROOT / "models"
DATA_RESULTS = DATA_ROOT / "results"
BPMN2_NAMESPACE = "http://b3mn.org/stencilset/bpmn2.0#"


anomaly_types = {
    0: 'out_of_order',
    1: 'superfluous_activity',
    2: 'missing_activity',
}