"""Sample/frame/slot accounting without an audio device or third-party package."""
import math


def stream_budget(rate, channels, sample_bytes, period_frames, periods, slots, slot_bits):
    values=(rate,channels,sample_bytes,period_frames,periods,slots,slot_bits)
    if any(type(x) is not int or x<=0 for x in values):
        raise ValueError('positive integer fields required')
    if slots<channels or slot_bits<sample_bytes*8:
        raise ValueError('slots must accommodate channels and sample width')
    return {'frame_bytes':channels*sample_bytes,'bytes_per_second':rate*channels*sample_bytes,
            'period_ms':1000*period_frames/rate,'buffer_ms':1000*period_frames*periods/rate,
            'bit_clock_hz':rate*slots*slot_bits}


def clock_drift(rate, ppm, seconds):
    if not all(math.isfinite(x) for x in (rate,ppm,seconds)) or rate<=0 or seconds<0:
        raise ValueError('finite positive rate and nonnegative duration required')
    error_seconds=ppm*1e-6*seconds
    return {'sample_difference':rate*error_seconds,'time_difference_ms':error_seconds*1000}
