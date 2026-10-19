"""Read actual GStreamer RGB buffers and measure age in pipeline running time."""
from __future__ import annotations

import argparse
import hashlib
import json
import time


def rgb_rows(data: bytes, width: int, height: int, stride: int, offset: int = 0) -> bytes:
    """Copy logical RGB pixels, excluding allocator row padding."""
    if any(type(value) is not int for value in (width, height, stride, offset)):
        raise ValueError("image geometry must be integer")
    if width <= 0 or height <= 0 or stride < width * 3 or offset < 0:
        raise ValueError("invalid RGB geometry")
    if offset + (height - 1) * stride + width * 3 > len(data):
        raise ValueError("buffer does not contain all logical rows")
    return b"".join(data[offset + row * stride:offset + row * stride + width * 3]
                    for row in range(height))


def gst_modules():
    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo
    except (ImportError, ValueError) as error:
        raise RuntimeError("install PyGObject, GStreamer introspection, and base plugins") from error
    Gst.init(None)
    for plugin in ("videotestsrc", "appsink"):
        if Gst.ElementFactory.find(plugin) is None:
            raise RuntimeError(f"missing GStreamer plugin: {plugin}")
    return Gst, GstVideo


def sink_policy(sink, drop_oldest: bool, maximum: int) -> str:
    sink.set_property("max-buffers", maximum)
    if sink.find_property("leaky-type") is not None:
        # GstAppLeakyType.DOWNSTREAM = 2, introduced with GStreamer 1.28.
        sink.set_property("leaky-type", 2 if drop_oldest else 0)
        return "leaky-type"
    sink.set_property("drop", drop_oldest)
    return "drop (pre-1.28 API)"


def run(frames: int = 30, consumer_ms: float = 40, drop_oldest: bool = True,
        live: bool = True, drain_after_eos: bool = False) -> dict:
    if type(frames) is not int or not 1 <= frames <= 300:
        raise ValueError("frames must be 1..300")
    if not isinstance(consumer_ms, (int, float)) or not 0 <= consumer_ms <= 100:
        raise ValueError("consumer_ms must be finite and 0..100")
    if drain_after_eos and (live or not drop_oldest):
        raise ValueError("drain-after-EOS fixture needs nonlive, dropping source")
    Gst, GstVideo = gst_modules()
    description = (f"videotestsrc num-buffers={frames} is-live={'true' if live else 'false'} pattern=black ! "
                   "video/x-raw,format=RGB,width=65,height=48,framerate=50/1 ! "
                   "appsink name=frames sync=false wait-on-eos=false")
    pipeline = Gst.parse_launch(description)
    sink = pipeline.get_by_name("frames")
    policy_api = sink_policy(sink, drop_oldest, 2)
    bus = pipeline.get_bus()
    rows = []
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("could not start pipeline")
        if drain_after_eos:
            message = bus.timed_pop_filtered(5 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
            if message is None: raise RuntimeError("source did not finish")
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                raise RuntimeError(f"GStreamer: {error}; {debug}")
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            sample = sink.emit("try-pull-sample", Gst.SECOND)
            if sample is None:
                message = bus.pop_filtered(Gst.MessageType.ERROR)
                if message is not None:
                    error, debug = message.parse_error()
                    raise RuntimeError(f"GStreamer: {error}; {debug}")
                if sink.get_property("eos"): break
                continue
            buffer = sample.get_buffer()
            info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
            data = buffer.extract_dup(0, buffer.get_size())
            pixels = rgb_rows(data, info.width, info.height, info.stride[0], info.offset[0])
            if buffer.pts == Gst.CLOCK_TIME_NONE:
                raise RuntimeError("missing presentation timestamp")
            running_pts = sample.get_segment().to_running_time(Gst.Format.TIME, buffer.pts)
            if running_pts == Gst.CLOCK_TIME_NONE:
                raise RuntimeError("timestamp outside segment")
            clock_now = pipeline.get_clock().get_time() - pipeline.get_base_time()
            rows.append({"frame_id": int(buffer.offset), "pts_ns": int(buffer.pts),
                         "running_pts_ns": int(running_pts),
                         "age_at_pull_ns": int(clock_now - running_pts) if live else None,
                         "width": info.width, "height": info.height, "stride": info.stride[0],
                         "allocated_bytes": buffer.get_size(), "logical_bytes": len(pixels),
                         "logical_sha256": hashlib.sha256(pixels).hexdigest(),
                         "all_black": not any(pixels)})
            if consumer_ms: time.sleep(consumer_ms / 1000)
        else:
            raise RuntimeError("bounded experiment timed out")
    finally:
        pipeline.set_state(Gst.State.NULL)
    ids = [row["frame_id"] for row in rows]
    if not rows or ids != sorted(set(ids)):
        raise RuntimeError("missing or out-of-order samples")
    return {"gstreamer": Gst.version_string(), "generated_frames": frames,
            "consumed_frames": len(rows), "omitted_frames": frames - len(rows),
            "consumer_ms": consumer_ms, "live": live, "drop_oldest": drop_oldest,
            "policy_api": policy_api, "samples": rows,
            "boundary": "CPU synthetic source; pipeline running time; no physical camera or GPU"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--consumer-ms", type=float, default=40)
    parser.add_argument("--keep-all", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.frames, args.consumer_ms, not args.keep_all), indent=2))


if __name__ == "__main__":
    main()
