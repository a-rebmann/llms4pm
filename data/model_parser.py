import json
import logging
from collections import deque
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

from const import DATA_DATASET, BPMN2_NAMESPACE

_logger = logging.getLogger(__name__)


def get_csv_paths(ds_root=DATA_DATASET) -> List[Path]:
    paths = sorted(ds_root.glob("*.csv"))
    assert len(paths) > 0, f"Could not find any csv in {ds_root.absolute()}, have you downloaded the dataset?"
    _logger.info("Found %d csvs", len(paths))
    return paths


def parse_csv_raw(csv_path: Path, **kwargs):
    df = (
        pd.read_csv(csv_path, dtype={"Type": "category", "Namespace": "category"}, **kwargs)
        .rename(columns=lambda s: s.replace(" ", "_").lower())
    )
    if df["model_id"].duplicated().any():
        _logger.warning("csv has %d duplicate model ids", df["model_id"].duplicated().sum())
    assert not df["namespace"].isna().any(), "csv has NA namespace entries, this should not happen."
    return df


def parse_model_metadata(csv_paths=None) -> pd.DataFrame:
    if csv_paths is None:
        csv_paths = get_csv_paths()
    _logger.info("Starting to parse %d cvs excluding model json", len(csv_paths))

    # exclude "Model JSON" column to speed up import and reduce memory usage
    dfs = [parse_csv_raw(p, usecols=lambda s: s != "Model JSON") for p in tqdm(csv_paths)]
    df = pd.concat(dfs)
    _logger.info("Parsed %d models", len(df))
    return df


def parse_models(csv_paths=None):
    if csv_paths is None:
        csv_paths = get_csv_paths()
    _logger.info("Starting to parse %d CSVs", len(csv_paths))
    dfs = [_parse_models(p) for p in tqdm(csv_paths)]
    df = pd.concat(dfs)
    return df


def _parse_models(csv_path):
    df = parse_csv_raw(csv_path)
    df_bpmn = df.query(f"namespace == '{BPMN2_NAMESPACE}'")
    return df_bpmn


class BpmnModelParser:
    def __init__(self, parse_outgoing=False, parse_parent=False):
        self.parse_outgoing = parse_outgoing
        self.parse_parent = parse_parent

    def parse_model_elements(self, csv_paths=None) -> pd.DataFrame:
        if csv_paths is None:
            csv_paths = get_csv_paths()
        _logger.info("Starting to parse %d cvs", len(csv_paths))
        dfs = [self._parse_bpmn_model_elements_csv(p) for p in tqdm(csv_paths)]
        df = pd.concat(dfs)
        return df

    def _parse_bpmn_model_elements_csv(self, csv_path: Path) -> pd.DataFrame:
        df = parse_csv_raw(csv_path)
        df_bpmn = df.query(f"namespace == '{BPMN2_NAMESPACE}'")
        model_dfs = [self._parse_df_row(t) for t in df_bpmn.reset_index().itertuples()]
        return (
            pd.concat(model_dfs)
            .set_index(["model_id", "element_id"])
            .astype({"category": "category"})  # convert column category to dtype categorical to save memory
        )

    def _parse_df_row(self, row_tuple):
        model_dict = json.loads(row_tuple.model_json)
        elements = self._get_elements_flat(model_dict)
        return (
            pd.DataFrame.from_records(elements)
            .assign(model_id=row_tuple.model_id)
        )

    def _get_elements_flat(self, model_dict) -> List[Dict[str, str]]:
        """
        Parses the recursive childShapes and produces a flat list of model elements with the most important attributes
        such as id, category, label, outgoing, and parent elements.
        """
        stack = deque([model_dict])
        elements_flat = []

        while len(stack) > 0:
            element = stack.pop()

            for c in element.get("childShapes", []):
                c["parent"] = element["resourceId"]
                stack.append(c)

            # don't append root as element
            if element["resourceId"] == model_dict["resourceId"]:
                continue

            # NOTE: it's possible to add other attributes here, such as the bounds of an element
            record = {
                "element_id": element["resourceId"],
                "category": element["stencil"].get("id") if "stencil" in element else None,
                "label": element["properties"].get("name"),
            }
            if self.parse_parent:
                record["parent"] = element.get("parent")
            if self.parse_outgoing:
                record["outgoing"] = [v for d in element.get("outgoing", []) for v in d.values()]

            elements_flat.append(record)

        return elements_flat
