#!/usr/bin/env python3

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a local PMLB checkout into an AutoMLBenchmark classification "
            "benchmark with AMLB-compatible train/test fold files."
        )
    )
    parser.add_argument(
        "--pmlb-root", required=True, help="Path to a local PMLB checkout."
    )
    parser.add_argument(
        "--output-benchmark",
        required=True,
        help="Where to write the generated AMLB benchmark YAML.",
    )
    parser.add_argument(
        "--output-data-dir",
        required=True,
        help="Directory where generated train/test CSV files should be stored.",
    )
    parser.add_argument(
        "--userdir",
        default=None,
        help=(
            "Optional AMLB userdir. If output-data-dir lives under this directory, "
            "the generated benchmark will use {user}-relative paths."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=1,
        help="Number of AMLB folds to generate per dataset. "
        "Use 1 for a single holdout split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout size used when folds=1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generated splits.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name to include. Repeat the flag to keep multiple datasets.",
    )
    parser.add_argument(
        "--dataset-list-file",
        default=None,
        help="Text file with one dataset name per line.",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Optional cap after dataset selection.",
    )
    parser.add_argument(
        "--max-classes",
        type=int,
        default=100,
        help="Heuristic upper bound for inferring numeric classification targets.",
    )
    parser.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        default="classification",
        help="Which PMLB task family to prepare.",
    )
    return parser.parse_args()


def load_requested_datasets(args: argparse.Namespace) -> List[str]:
    requested = list(args.dataset)
    if args.dataset_list_file:
        lines = Path(args.dataset_list_file).read_text(encoding="utf-8").splitlines()
        requested.extend(
            line.strip()
            for line in lines
            if line.strip() and not line.startswith("#")
        )
    seen = set()
    ordered = []
    for name in requested:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def discover_dataset_files(pmlb_root: Path) -> Dict[str, Path]:
    datasets_root = (
        pmlb_root / "datasets" if (pmlb_root / "datasets").is_dir() else pmlb_root
    )
    dataset_files: Dict[str, Path] = {}
    for dataset_dir in sorted(p for p in datasets_root.iterdir() if p.is_dir()):
        candidates = []
        for suffix in (".tsv.gz", ".csv.gz", ".tsv", ".csv"):
            top_level = dataset_dir / f"{dataset_dir.name}{suffix}"
            if top_level.is_file():
                candidates.append(top_level)
        if not candidates:
            continue
        preferred = next(
            (p for p in candidates if p.name.endswith(".tsv.gz")), candidates[0]
        )
        dataset_files[dataset_dir.name] = preferred
    return dataset_files


def infer_task_from_metadata(dataset_file: Path) -> Optional[str]:
    for metadata_name in ("metadata.yaml", "metadata.yml", "metadata.json"):
        metadata_path = dataset_file.parent / metadata_name
        if not metadata_path.exists():
            continue
        if metadata_path.suffix == ".json":
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            for key in ("task", "task_type", "problem_type"):
                value = data.get(key)
                if not isinstance(value, str):
                    continue
                normalized = value.lower()
                if "class" in normalized:
                    return "classification"
                if "regress" in normalized:
                    return "regression"
        else:
            for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                if key.strip() not in {"task", "task_type", "problem_type"}:
                    continue
                normalized = value.strip().strip("'\"").lower()
                if "class" in normalized:
                    return "classification"
                if "regress" in normalized:
                    return "regression"
    return None


def separator_for(path: Path) -> str:
    return "\t" if ".tsv" in path.name else ","


def assert_not_lfs_pointer(path: Path) -> None:
    with path.open("rb") as fh:
        head = fh.read(128)
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise SystemExit(
            f"Dataset file `{path}` is a Git LFS "
            "pointer. Clone PMLB with Git LFS "
            "enabled or use a downloaded archive "
            "that contains the actual data files."
        )


def is_integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return False
    rounded = np.round(numeric.to_numpy(dtype=float))
    return np.allclose(numeric.to_numpy(dtype=float), rounded)


def infer_target_column(columns: List[str]) -> str:
    lowered = {col.lower(): col for col in columns}
    for candidate in (
        "target",
        "class",
        "label",
        "y",
        "output",
        "response",
        "species",
    ):
        if candidate in lowered:
            return lowered[candidate]
    return columns[-1]


def read_target_column(dataset_file: Path) -> pd.Series:
    preview = pd.read_csv(
        dataset_file,
        sep=separator_for(dataset_file),
        compression="infer",
        nrows=0,
    )
    if len(preview.columns) == 0:
        raise SystemExit(f"Dataset `{dataset_file}` has no columns.")
    target_column = infer_target_column(list(preview.columns))
    return pd.read_csv(
        dataset_file,
        sep=separator_for(dataset_file),
        compression="infer",
        usecols=[target_column],
    )[target_column].dropna()


def is_classification_dataset(dataset_file: Path, max_classes: int) -> bool:
    assert_not_lfs_pointer(dataset_file)
    task_type = infer_task_from_metadata(dataset_file)
    if task_type is not None:
        return task_type == "classification"

    target = read_target_column(dataset_file)
    if target.empty:
        return False
    if target.dtype.kind in {"O", "b"}:
        return True
    unique_values = target.nunique(dropna=True)
    return 2 <= unique_values <= max_classes and is_integer_like(target)


def is_regression_dataset(dataset_file: Path, max_classes: int) -> bool:
    assert_not_lfs_pointer(dataset_file)
    task_type = infer_task_from_metadata(dataset_file)
    if task_type is not None:
        return task_type == "regression"
    target = read_target_column(dataset_file)
    if target.empty:
        return False
    if target.dtype.kind in {"O", "b"}:
        return False
    unique_values = target.nunique(dropna=True)
    return not (2 <= unique_values <= max_classes and is_integer_like(target))


def select_classification_datasets(
    dataset_files: Dict[str, Path], requested: List[str], max_classes: int
) -> Dict[str, Path]:
    if requested:
        missing = [name for name in requested if name not in dataset_files]
        if missing:
            missing_str = ", ".join(missing)
            raise SystemExit(
                f"Requested datasets not found in PMLB checkout: {missing_str}"
            )
        return {name: dataset_files[name] for name in requested}

    if importlib.util.find_spec("pmlb") is not None:
        from pmlb import classification_dataset_names

        names = [
            name for name in classification_dataset_names if name in dataset_files
        ]
        return {name: dataset_files[name] for name in names}

    selected = {}
    for name, dataset_file in dataset_files.items():
        if is_classification_dataset(dataset_file, max_classes=max_classes):
            selected[name] = dataset_file
    return selected


def select_regression_datasets(
    dataset_files: Dict[str, Path], requested: List[str], max_classes: int
) -> Dict[str, Path]:
    if requested:
        missing = [name for name in requested if name not in dataset_files]
        if missing:
            missing_str = ", ".join(missing)
            raise SystemExit(
                f"Requested datasets not found in PMLB checkout: {missing_str}"
            )
        selected = {}
        for name in requested:
            dataset_file = dataset_files[name]
            if not is_regression_dataset(dataset_file, max_classes=max_classes):
                raise SystemExit(
                    f"Requested dataset `{name}` is not a regression dataset."
                )
            selected[name] = dataset_file
        return selected

    if importlib.util.find_spec("pmlb") is not None:
        from pmlb import regression_dataset_names

        names = [name for name in regression_dataset_names if name in dataset_files]
        return {name: dataset_files[name] for name in names}

    selected = {}
    for name, dataset_file in dataset_files.items():
        if is_regression_dataset(dataset_file, max_classes=max_classes):
            selected[name] = dataset_file
    return selected


def train_test_names(dataset_name: str, fold: int, folds: int) -> Tuple[str, str]:
    if folds == 1:
        return f"{dataset_name}_train.csv", f"{dataset_name}_test.csv"
    return f"{dataset_name}_train_{fold}.csv", f"{dataset_name}_test_{fold}.csv"


def dataset_path_for_yaml(dataset_dir: Path, userdir: Optional[Path]) -> str:
    if userdir is not None:
        try:
            relative = dataset_dir.relative_to(userdir)
            return "{user}/" + relative.as_posix()
        except ValueError:
            pass
    return str(dataset_dir)


def yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def render_benchmark_yaml(entries: List[dict]) -> str:
    lines = ["---", ""]
    for entry in entries:
        lines.append(f"- name: {yaml_quote(entry['name'])}")
        lines.append("  dataset:")
        lines.append(f"    path: {yaml_quote(entry['dataset']['path'])}")
        lines.append(f"    target: {yaml_quote(entry['dataset']['target'])}")
        lines.append(f"    type: {yaml_quote(entry['dataset']['type'])}")
        lines.append(f"  folds: {entry['folds']}")
        lines.append(f"  description: {yaml_quote(entry['description'])}")
        lines.append("")
    return "\n".join(lines)


def write_dataset_splits(
    dataset_name: str,
    dataset_file: Path,
    dataset_output_dir: Path,
    folds: int,
    test_size: float,
    seed: int,
    task_type: str,
) -> str:
    assert_not_lfs_pointer(dataset_file)
    data = pd.read_csv(
        dataset_file,
        sep=separator_for(dataset_file),
        compression="infer",
    )
    if len(data.columns) == 0:
        raise SystemExit(f"Dataset `{dataset_name}` has no columns.")
    target_column = infer_target_column(list(data.columns))

    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    if target_column != "target":
        data = data.rename(columns={target_column: "target"})
    X = data.drop(columns=["target"])
    y = data["target"]
    if task_type == "regression":
        problem_type = "regression"
    else:
        problem_type = "binary" if y.nunique(dropna=True) == 2 else "multiclass"

    if folds == 1:
        if task_type == "regression":
            rng = np.random.RandomState(seed)
            indices = np.arange(len(data))
            rng.shuffle(indices)
            test_count = max(1, int(round(len(indices) * test_size)))
            test_idx = indices[:test_count]
            train_idx = indices[test_count:]
            splits = [(train_idx, test_idx)]
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed
            )
            splits = splitter.split(X, y)
    else:
        if task_type == "regression":
            indices = np.arange(len(data))
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)
            folds_indices = np.array_split(indices, folds)
            splits = []
            for fold in range(folds):
                test_idx = folds_indices[fold]
                train_idx = np.concatenate(
                    [folds_indices[i] for i in range(folds) if i != fold]
                )
                splits.append((train_idx, test_idx))
        else:
            splitter = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=seed
            )
            splits = splitter.split(X, y)

    for fold, (train_idx, test_idx) in enumerate(splits):
        train_name, test_name = train_test_names(dataset_name, fold, folds)
        data.iloc[train_idx].to_csv(dataset_output_dir / train_name, index=False)
        data.iloc[test_idx].to_csv(dataset_output_dir / test_name, index=False)
    return problem_type


def main() -> None:
    args = parse_args()
    if args.folds < 1:
        raise SystemExit("--folds must be >= 1")
    if args.folds == 1 and not 0.0 < args.test_size < 1.0:
        raise SystemExit("--test-size must be between 0 and 1 when folds=1")

    pmlb_root = Path(args.pmlb_root).expanduser().resolve()
    output_benchmark = Path(args.output_benchmark).expanduser().resolve()
    output_data_dir = Path(args.output_data_dir).expanduser().resolve()
    userdir = Path(args.userdir).expanduser().resolve() if args.userdir else None

    dataset_files = discover_dataset_files(pmlb_root)
    if not dataset_files:
        raise SystemExit(f"No dataset files were found under {pmlb_root}")

    requested = load_requested_datasets(args)
    if args.task_type == "classification":
        selected = select_classification_datasets(
            dataset_files, requested, args.max_classes
        )
    else:
        selected = select_regression_datasets(
            dataset_files, requested, args.max_classes
        )
    selected_names = sorted(selected)
    if args.max_datasets is not None:
        selected_names = selected_names[: args.max_datasets]
    if not selected_names:
        raise SystemExit("No classification datasets were selected.")

    benchmark_entries = []
    output_benchmark.parent.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in selected_names:
        dataset_file = selected[dataset_name]
        dataset_dir = output_data_dir / dataset_name
        problem_type = write_dataset_splits(
            dataset_name=dataset_name,
            dataset_file=dataset_file,
            dataset_output_dir=dataset_dir,
            folds=args.folds,
            test_size=args.test_size,
            seed=args.seed,
            task_type=args.task_type,
        )
        benchmark_entries.append(
            {
                "name": dataset_name,
                "dataset": {
                    "path": dataset_path_for_yaml(dataset_dir, userdir),
                    "target": "target",
                    "type": problem_type,
                },
                "folds": args.folds,
                "description": f"PMLB {args.task_type} "
                f"dataset prepared from {dataset_file.name}",
            }
        )

    output_benchmark.write_text(
        render_benchmark_yaml(benchmark_entries), encoding="utf-8"
    )
    print(f"Wrote benchmark definition to {output_benchmark}")
    print(f"Wrote dataset splits under {output_data_dir}")
    print(f"Prepared {len(benchmark_entries)} {args.task_type} datasets.")


if __name__ == "__main__":
    main()
