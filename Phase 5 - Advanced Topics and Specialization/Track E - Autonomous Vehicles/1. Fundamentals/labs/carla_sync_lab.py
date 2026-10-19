"""Optional CARLA 0.9.16 local sensor/frame exercise. Not executed in course review.

Requires an otherwise unused local CARLA server. Creates only its own actors,
ticks the simulation for a bounded period, then restores world settings.
"""
import argparse
import json
from pathlib import Path
import queue
import time


def matching_frame(images, expected, timeout=5.):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline-time.monotonic()
        if remaining <= 0: raise TimeoutError(f'camera frame {expected} did not arrive')
        try: image=images.get(timeout=remaining)
        except queue.Empty as exc: raise TimeoutError(f'camera frame {expected} did not arrive') from exc
        if image.frame == expected: return image
        if image.frame > expected: raise RuntimeError(f'missing camera frame {expected}; got {image.frame}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ticks',type=int,default=40)
    parser.add_argument('--output',type=Path,default=Path('/tmp/course-carla-frames.json'))
    args=parser.parse_args()
    if not 1 <= args.ticks <= 200: parser.error('ticks must be in [1,200]')
    import carla
    client=carla.Client('127.0.0.1',2000);client.set_timeout(10.)
    client_version,server_version=client.get_client_version(),client.get_server_version()
    if client_version != server_version or not client_version.startswith('0.9.16'):
        raise RuntimeError(f'requires matching 0.9.16 client/server; got {client_version}/{server_version}')
    world=client.get_world();original=world.get_settings()
    if original.synchronous_mode:
        raise RuntimeError('server is already synchronous; use an unused server owned by this lab')
    if original.no_rendering_mode:
        raise RuntimeError('RGB sensor requires rendering; restart the lab server with rendering available')
    settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05
    settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=10
    vehicle=camera=None;records=[];images=queue.Queue(maxsize=8)
    def receive(image):
        try: images.put_nowait(image)
        except queue.Full: pass  # bounded storage; missing frame is reported by matching_frame
    try:
        world.apply_settings(settings)
        blueprint=world.get_blueprint_library().find('vehicle.tesla.model3')
        for spawn in world.get_map().get_spawn_points():
            vehicle=world.try_spawn_actor(blueprint,spawn)
            if vehicle is not None: break
        if vehicle is None: raise RuntimeError('no available spawn point')
        sensor=world.get_blueprint_library().find('sensor.camera.rgb')
        sensor.set_attribute('image_size_x','320');sensor.set_attribute('image_size_y','240')
        sensor.set_attribute('sensor_tick','0.0')
        camera=world.spawn_actor(sensor,carla.Transform(carla.Location(x=1.5,z=2.4)),attach_to=vehicle)
        camera.listen(receive)
        # Stationary frame transport exercise, not a driving/vehicle-dynamics result.
        vehicle.apply_control(carla.VehicleControl(throttle=0.,brake=1.,steer=0.))
        for _ in range(args.ticks):
            frame=world.tick(10.);snapshot=world.get_snapshot();image=matching_frame(images,frame)
            if snapshot.frame != frame or abs(image.timestamp-snapshot.timestamp.elapsed_seconds)>1e-6:
                raise RuntimeError('sensor/world frame or simulation timestamp mismatch')
            if len(image.raw_data) != 320*240*4: raise RuntimeError('unexpected BGRA byte count')
            records.append({'frame':frame,'sensor_timestamp_s':image.timestamp,
                            'world_timestamp_s':snapshot.timestamp.elapsed_seconds,
                            'bgra_bytes':len(image.raw_data)})
    finally:
        try:
            if camera is not None:
                camera.stop();camera.destroy()
        finally:
            try:
                if vehicle is not None:vehicle.destroy()
            finally:world.apply_settings(original)
    report={'client':client_version,'server':server_version,'scope':'stationary local sensor synchronization',
            'fixed_delta_s':.05,'frames':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: {len(records)} matched frames; saved {args.output}')


if __name__=='__main__':main()
