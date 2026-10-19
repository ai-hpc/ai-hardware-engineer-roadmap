"""Standard-library input contract shared by the optional NumPy/OpenCV lab."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Measurement:
    sample_time: float
    arrival_time: float
    kind: str
    value: float
    variance: float
    sequence: int

    def __post_init__(self):
        if not all(math.isfinite(x) for x in
                   (self.sample_time, self.arrival_time, self.value, self.variance)):
            raise ValueError('measurements must be finite')
        if self.sample_time < 0 or self.arrival_time < self.sample_time:
            raise ValueError('require 0 <= sample_time <= arrival_time in one clock')
        if self.variance <= 0 or self.kind not in ('camera', 'wheel'):
            raise ValueError('positive variance and known sensor kind required')
        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError('sequence must be a nonnegative integer')


def visible_measurements(measurements, now):
    """A measurement may affect only outputs at/after its arrival."""
    if not math.isfinite(now) or now < 0:
        raise ValueError('finite nonnegative current time required')
    return sorted((m for m in measurements if m.arrival_time <= now + 1e-10),
                  key=lambda m: (m.sample_time, m.sequence))
