"""Training-data layer: bridges cleaned GPS (preprocessing) + road graph (roadgraph)
into model-ready batches and the P3 physical-constraint reward.

Contains NO model code and runs NO training loop -- only the dataset / candidate
retrieval / reward machinery the Stage 1-3 trainers consume.
"""

from dataset.config import (
    CITY_OF_SOURCE, CITY_UTM_CRS, RetrievalConfig, RewardConfig, SequenceConfig,
)

__all__ = [
    "CITY_OF_SOURCE", "CITY_UTM_CRS",
    "RetrievalConfig", "RewardConfig", "SequenceConfig",
]
