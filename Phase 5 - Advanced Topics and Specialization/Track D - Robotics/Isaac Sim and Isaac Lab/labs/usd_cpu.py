#!/usr/bin/env python3
"""Actual OpenUSD composition/units checks, without Isaac Sim or any asset download."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile


def experiment():
    from pxr import Gf, Sdf, Usd, UsdGeom

    with tempfile.TemporaryDirectory(prefix="course-usd-") as work:
        directory = Path(work)
        asset = Usd.Stage.CreateNew(str(directory / "asset.usda"))
        robot = UsdGeom.Xform.Define(asset, "/Robot")
        asset.SetDefaultPrim(robot.GetPrim())
        cube = UsdGeom.Cube.Define(asset, "/Robot/Body")
        cube.CreateSizeAttr(2.0)
        asset.GetRootLayer().Save()

        stage = Usd.Stage.CreateNew(str(directory / "scene.usda"))
        prim = stage.DefinePrim("/World/Robot", "Xform")
        prim.GetPayloads().AddPayload("asset.usda")
        body = stage.GetPrimAtPath("/World/Robot/Body")
        assert body.GetAttribute("size").Get() == 2
        body.GetAttribute("size").Set(3.0)  # local opinion overrides the payload.
        assert body.GetAttribute("size").Get() == 3
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            body.GetAttribute("size").Set(4.0)
        assert body.GetAttribute("size").Get() == 4
        stage.GetRootLayer().Save()
        reopened = Usd.Stage.Open(str(directory / "scene.usda"))
        assert reopened.GetPrimAtPath("/World/Robot/Body").GetAttribute("size").Get() == 3
        flattened = Usd.Stage.Open(stage.Flatten())
        assert flattened.GetPrimAtPath("/World/Robot/Body").GetAttribute("size").Get() == 4

        unloaded = Usd.Stage.Open(str(directory / "scene.usda"), Usd.Stage.LoadNone)
        host = unloaded.GetPrimAtPath("/World/Robot")
        assert host and host.HasPayload() and not host.IsLoaded()
        # A local override can keep a descendant prim present even when its payload is unloaded.
        local_body = unloaded.GetPrimAtPath("/World/Robot/Body")
        assert local_body and local_body.GetAttribute("size").Get() == 3
        assert not local_body.GetTypeName()  # Cube type was contributed by the unloaded payload.
        unloaded.Load("/World/Robot")
        assert local_body.GetTypeName() == "Cube"

        # Stage metadata does not rewrite xform numbers or rescale referenced geometry.
        frame = UsdGeom.Xform.Define(stage, "/World/Frame")
        operation = frame.AddTranslateOp()
        operation.Set(Gf.Vec3d(100, 0, 0))
        UsdGeom.SetStageMetersPerUnit(stage, .01)
        before = operation.Get()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        assert operation.Get() == before == Gf.Vec3d(100, 0, 0)

        return {"scope": "actual OpenUSD CPU composition; no physics simulation or renderer",
                "usd_core": importlib.metadata.version("usd-core"),
                "opinion_values": {"asset": 2, "local": 3, "session": 4,
                                   "reopened_root": 3, "flattened_stage": 4},
                "unloaded_payload_host_present": True,
                "local_override_descendant_present_without_payload_type": True,
                "translation_numbers_after_units_metadata_change": list(operation.Get()),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(experiment(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report, end="")
