from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Feature(object):
    name: str
    type: Any
    converter: Optional[Callable] = None


@dataclass
class SampleItem(object):
    track: str
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratorOptions(object):
    n_prev_likes: int = 10
    n_fav_artists: int = 10
    sample_size: int = 100
    for_train: bool = True


@dataclass
class Object(object):
    user_id: int
    prev_likes: List[str] = field(default_factory=list)
    prev_artists: List[str] = field(default_factory=list)
    fav_artists: List[str] = field(default_factory=list)
    fav_artists_likes: List[int] = field(default_factory=list)
    cur_track_id: str = ''
    cur_artist_id: str = ''
    sampler_rank: float = 0.0
    add_features: Dict[str, Any] = field(default_factory=dict)
    like: Optional[int] = None
    rank: Optional[float] = None
