import os

from typing import Optional, Tuple, Dict, Any, Callable, Union
from collections.abc import Sequence, Iterator
from copy import copy
import shutil

from dataclasses import dataclass
from pathlib import Path
import logging

from multiprocessing import Pool

import numpy as np

from streaming import MDSWriter, JSONWriter

from simple_parsing import field

from datatools.utils import Subset, merge_index_recursively

logger = logging.getLogger(__name__)


@dataclass
class ProcessOptions:
    """Options for process function"""

    # Number of workers to use
    num_proc: Optional[int] = field(alias=["-w", "--num_workers"], default=None)

    # Range of rows to process
    index_range: Optional[Tuple[int, int]] = None
    index_path: Optional[Path] = None
    indices: Optional[np.ndarray] = field(cmd=False, default=None)
    sort_index: bool = False

    job_id: Optional[int] = None    # Job id
    num_jobs: Optional[int] = None  # Number of jobs

    # Read slurm job array environment variables. Gets overridden by job_id/num_jobs
    slurm_array: bool = False

    compression: Optional[str] = None  # Compress output files
    jsonl: bool = False                # Write JSONL files

    # Specify column_types like "input_ids=ndarray:uint32,domain=str"
    column_types: str = field(alias=["-c"], default=None)
    columns: Dict[str, str] = field(cmd=False, default=None)

    overwrite: bool = False

    def __post_init__(self):
        if self.slurm_array and self.job_id is None:
            self.job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
            logger.warning(f"Using SLURM array environment variable: job_id={self.job_id}")
        if self.slurm_array and self.num_jobs is None:
            self.num_jobs = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
            logger.warning(f"Using SLURM array environment variable: num_jobs={self.num_jobs}")
        if self.column_types is not None and self.columns is None:
            self.columns = dict([col.split("=") for col in self.column_types.split(",")])


def subset_output_path(output_path: Path, subset: str, process_id: int, options: ProcessOptions) -> str:
    parts = []

    if options.job_id is not None and options.num_jobs is not None:
        num_jobs = str(options.num_jobs)
        parts.append(f"job{options.job_id:0{len(num_jobs)}}-{num_jobs}")
    if options.num_proc is not None:
        num_proc = str(options.num_proc)
        parts.append(f"proc{process_id:0{len(num_proc)}}-{num_proc}")

    return str(output_path / subset / ("_".join(parts) if parts else ""))


def infer_columns(item):
    columns = {}
    for key, value in item.items():
        if isinstance(value, np.ndarray):
            columns[key] = f"ndarray:{value.dtype}"
        elif isinstance(value, np.number):
            columns[key] = str(value.dtype)
        elif isinstance(value, str):
            columns[key] = "str"
        elif isinstance(value, int):
            columns[key] = "int"
        elif isinstance(value, float):
            columns[key] = "float"
        else:
            columns[key] = "pkl"
    return columns


def identity_fn(dataset, *_):
    for i in range(len(dataset)):
        yield Path(), dataset[i]


def write_process_(args):
    dataset, indices, process_fn, output_path, options, process_id = args

    dataset = Subset.shard(dataset, process_id, options.num_proc or 1)
    indices = Subset.shard(indices, process_id, options.num_proc or 1)

    writer_cls = JSONWriter if options.jsonl else MDSWriter
    writers = {}

    try:
        for result in process_fn(dataset, indices, process_id):
            if isinstance(result, tuple):
                subset, item = result
                subset = Path(subset if subset is not None else "")
            else:
                item = result
                subset = Path("")


            if subset not in writers:
                if options.columns is None:
                    columns = infer_columns(item)
                    logger.warning(f"Inferred columns \"{subset}\": {columns}")
                else:
                    columns = options.columns

                writers[subset] = writer_cls(
                    columns=columns,
                    out=subset_output_path(output_path, subset, process_id, options),
                    compression=options.compression)
            writers[subset].write(item)
    finally:
        for writer in writers.values():
            writer.finish()


def load_indices(options):
    indices = None
    if options.indices is not None:
        assert options.index_path is None, "Cannot specify both indices and index_path"
        assert options.index_range is None, "Cannot specify both indices and index_range"

        indices = options.indices

    if options.index_path is not None:
        assert options.index_range is None, "Cannot specify both index_path and index_range"

        indices = np.load(options.index_path)
        logger.warning(f"Loaded {len(indices)} indices from {options.index_path}")

    if indices is not None and options.sort_index:
        indices = np.sort(indices)

    if options.index_range is not None:
        logger.warning(f"Using indices from {options.index_range[0]} to {options.index_range[1]}")
        indices = range(*options.index_range)

    return indices


def process(dataset: Sequence,
            process_fn: Callable[[Subset, Subset, int], Iterator[Tuple[Path, Dict[str, Any]]]],
            output_path: Union[Path, str],
            options: Optional[ProcessOptions] = None):
    options = copy(options)

    output_path = Path(output_path)

    if options.overwrite and output_path.exists():
        assert options.num_jobs is None or options.num_jobs == 1, "overwrite is incompatible with multiple jobs"
        shutil.rmtree(output_path)
        logger.warning(f"Removed existing output directory: {output_path}")

    indices = load_indices(options)
    if indices is not None:
        dataset = Subset(dataset, indices)
        logger.warning(f"Selected {len(dataset)} indices")
    else:
        indices = range(len(dataset))

    if options.job_id is not None and options.num_jobs is not None:
        if len(dataset) < options.num_jobs:
            options.num_jobs = len(dataset)
            logger.warning(f"Setting num_jobs={options.num_jobs} to match the dataset size")

        dataset = Subset.shard(dataset, options.job_id, options.num_jobs)
        indices = Subset.shard(indices, options.job_id, options.num_jobs)

    if options.num_proc and len(dataset) < options.num_proc:
        logger.warning(f"Setting num_proc={len(dataset)} to match the dataset size")
        options.num_proc = len(dataset)

    process_args = [(dataset, indices, process_fn, output_path, options, i) for i in range(options.num_proc or 1)]
    if options.num_proc is None or options.num_proc == 1:
        write_process_(process_args[0])
    else:
        with Pool(options.num_proc) as pool:
            pool.map(write_process_, process_args)

    # This gets executed by all jobs but each job will update all index.json
    merge_index_recursively(output_path)
