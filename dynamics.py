import math
import os
from collections import defaultdict, namedtuple
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import torch

sample_identifier = Union[int, str]

@dataclass
class DMRecord:
    """
    Class for holding info around a data map update for a single sample
    """
    guid: sample_identifier
