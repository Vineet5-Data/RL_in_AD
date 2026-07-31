"""Road-network graph construction (P1b data).

Turns OSM drivable roads for a city bbox into the segment-as-node line graph the
P1b semantic vectorizer / GAT and the P2 RL agent consume: each road segment is a
node, segments sharing an intersection are connected by a turn edge.
"""

from roadgraph.config import GraphConfig, CITY_BBOX, TEST_TILES, SEGMENT_COLUMNS

__all__ = ["GraphConfig", "CITY_BBOX", "TEST_TILES", "SEGMENT_COLUMNS"]
