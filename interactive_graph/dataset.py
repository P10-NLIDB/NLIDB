from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class DataArguments:
    dataset: str = field(
        metadata={"help": "The dataset to be used. Choose between ``spider``, ``cosql``, ``ambiQT``, or ``cosql+spider``."},
    )
    dataset_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "spider": "./datasets/spider",
            "sparc": "./datasets/sparc",
            "cosql": "./datasets/cosql",
            "ambiQT": "./datasets/ambiQT", # Only ambiQT implemented
        },
        metadata={"help": "Paths of the dataset modules."},
    )
    metric_config: str = field(
        default="exact_match",
        metadata={"help": "Choose between ``exact_match``, ``test_suite``, or ``both``."},
    )

    # Not yet implemented:
    metric_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "spider": "./metrics/spider",
            "sparc": "./metrics/sparc",
            "cosql": "./metrics/cosql",
            "ambiQT": "metrics/ambiQT", # Only ambbiQT implemented
        },
        metadata={"help": "Paths of the metric modules."},
    )
    test_suite_db_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test-suite databases."})
    data_config_file : Optional[str] = field(
        default=None,
        metadata={"help": "Path to data configuration file (specifying the database splits)"}
    )
    test_sections : Optional[List[str]] = field(
        default=None,
        metadata={"help": "Sections from the data config to use for testing"}
    )
    data_base_dir : Optional[str] = field(
        default="./dataset_files/",
        metadata={"help": "Base path to the lge relation dataset."})
    split_dataset : Optional[str] = field(
        default="",
        metadata={"help": "The dataset name after spliting."})