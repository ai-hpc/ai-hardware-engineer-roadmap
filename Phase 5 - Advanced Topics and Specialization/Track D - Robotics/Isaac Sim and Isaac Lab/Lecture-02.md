# Lecture 02: USD for Roboticists

## Overview

Every robot, table, light, camera and physics setting in Isaac Sim is USD. You can get through Lecture 01 without knowing that. You can't get much further. An imported arm that is 100× too large, a gripper whose friction setting "doesn't take", a scene of 64 robots that won't fit in VRAM, a training loop that crawls because it reads poses through the wrong layer: all of these are USD problems before they are physics or GPU problems.

This lecture covers the parts of OpenUSD that matter to a robotics engineer: the data model, how layers and composition arcs decide which value wins, the arcs real robot assets use (references, payloads, variants, instancing), physics schemas, units, and the three data pathways (USD, Fabric, physics tensors) that decide how fast you can read the scene back. You will take apart NVIDIA's hosted Franka asset, build a small work cell, and measure what instancing and the data backend actually cost on your card.

By the end you should be able to:

* explain prims, attributes, relationships and metadata, and read a `.usda` file
* predict which opinion wins when the same attribute is set in several layers or arcs (LIVERPS)
* use references, payloads, variants and instancing on purpose, and say what each costs in memory and load time
* tell typed schemas from applied API schemas, and find the articulation root, rigid bodies and joints in any asset
* catch unit, up-axis and xform-op bugs before they become physics bugs
* choose between the `usd`, `usdrt`, `fabric` and `tensor` backends, and explain why reading poses through USD in a loop is slow

---

## 1. Why it matters: the scene file is a performance decision

| Symptom | USD cause | Section |
|---|---|---|
| Imported robot is huge, tiny, or lying on its side | `metersPerUnit` / `upAxis` mismatch, or a scale buried in an xform op | 5 |
| You set friction or a drive gain, and nothing changes | Your edit is a weaker opinion than one in another layer | 2.3 |
| 64 copies of a robot use far more memory than one | Meshes are not instanced, so each copy is unique geometry | 3.4 |
| Opening a large scene takes minutes | Everything is loaded eagerly, with no payloads | 3.2 |
| Pose reads dominate step time, or come back stale | Reading through the `usd` backend, in a Python loop, after GPU physics | 6 |

USD came from film pipelines, where hundreds of artists edit one shot without overwriting each other. That is where its strengths come from (non-destructive layering, asset reuse). It is also where its one big weakness for robotics comes from: USD is built for authoring, not for being read thousands of times a second.

---

## 2. Mental model: prims, properties, stages, layers

### 2.1 The data model

A **stage** is a composed scene graph of **prims**, each at a path like `/panda/panda_link1`. A prim has:

* a **type** (`Xform`, `Mesh`, `Cube`, `PhysicsRevoluteJoint`, ...) or no type at all (an `over` or a plain `def`)
* **attributes**: typed, possibly time-sampled values (`double3 xformOp:translate`, `float physics:lowerLimit`)
* **relationships**: pointers to other prims (`rel physics:body0 = </panda/panda_link0>`, material bindings)
* **metadata**: data about the prim itself (`instanceable = true`, `kind`, `apiSchemas`, variant selections)

Here is a real joint from the Isaac Sim 6.1 Franka asset (`payloads/Physics/physics.usda`), trimmed. It shows all four:

```usda
over "panda" ( prepend apiSchemas = ["PhysicsArticulationRootAPI"] )      # "over": add opinions, define nothing
{
  over "panda_link0" ( prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"] )
  {
    float physics:mass = 2.8142028
    def PhysicsRevoluteJoint "panda_joint1" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular", "PhysicsJointStateAPI:angular"]   # metadata
    )
    {
        uniform token physics:axis = "X"                        # attribute
        prepend rel physics:body0 = </panda/panda_link0>        # relationship
        prepend rel physics:body1 = </panda/panda_link1>
        float physics:lowerLimit = -166.00307                   # degrees: USD angles are degrees
        float physics:upperLimit = 166.00307
        float drive:angular:physics:stiffness = 400             # the PD drive (Lecture 04)
    }
  }
}
```

Note the `over`. This layer doesn't define the links. It only adds physics opinions on top of prims that `base.usda` defines, and that split is the whole point of layering. `.usda` is text and diffs cleanly in git. `.usd` / `.usdc` is the binary crate format, faster to load and much smaller for meshes. NVIDIA's assets use text for structure and physics, binary for geometry.

### 2.2 Layers, the stage, and edit targets

A **layer** is one file's worth of scene description (opinions). A stage is the result of composing many layers:

```text
  session layer        (in memory, strongest, never saved by default: UI and debug state)
  root layer           (the file you opened)
    └─ sublayers       (stacked under the root; earlier in the list = stronger)
       + every file pulled in by references, payloads, variants, inherits, ...
```

* The **root layer** plus its **sublayers** form the root **layer stack**. Sublayering is how the 6.1 Franka asset stacks `physx.usda` on top of a neutral `physics.usda`.
* The **session layer** is the strongest layer and is not saved with the stage. Use it for temporary overrides (debug colors, a test gain) that must never leak into the asset.
* The **edit target** is the layer your writes go to, by default the root layer. `with Usd.EditContext(stage, stage.GetSessionLayer()): ...` redirects writes temporarily, so the change is visible now but never saved.

### 2.3 Opinions and strength ordering, simply

Every layer can hold an **opinion** about an attribute. When several disagree, the strongest one wins. OpenUSD's rule is **LIVERPS**: **L**ocal, **I**nherits, **V**ariantSets, r**E**locates, **R**eferences, **P**ayloads, **S**pecializes. Older material says LIVRPS; relocates were added to the acronym later. The idea is the same.

| Rank | Arc | What it means for you |
|---|---|---|
| 1 (strongest) | **Local** | Opinions in your own layer stack: root, its sublayers, and the session layer. Your scene file beats the asset. |
| 2-4 | **Inherits**, **VariantSets**, **Relocates** | The selected variant's opinions. Inherits and relocates are rare in robot assets. |
| 5-6 | **References**, **Payloads** | The asset you pulled in (payloads are loadable on demand). |
| 7 (weakest) | **Specializes** | Fallback defaults. Rare. |

Two rules of thumb get you through nearly every robotics case:

1. **The scene that uses an asset beats the asset.** An `over` on `/World/robot/panda_link3` in your work-cell file overrides the referenced Franka's value without touching the Franka file. That is how you tune one robot without forking the asset.
2. **Inside a layer stack, a layer beats its sublayers, and earlier sublayers beat later ones.** The Franka's `physx.usda` sublayers `physics.usda`, so its PhysX opinions sit on top of the neutral ones.

The classic "my change doesn't take" bug is the reverse of rule 1. You edit the asset file, but the scene (or a stronger sublayer, or the session layer) already has an opinion on that attribute. To debug it, ask USD where the value comes from. `attr.GetPropertyStack()` lists every spec contributing to an attribute, strongest first.

---

## 3. Composition arcs robots actually use

### 3.1 References: asset reuse

A **reference** grafts another file's prim tree under one of your prims. `stage_utils.add_reference_to_stage(usd_path=..., path=...)` (Lecture 01) authors exactly this. Ten references to one robot file share the file on disk and in the layer cache, but USD composes **ten independent prim trees**, so you can override each one separately. That independence is what references are for. It is also why ten references are not automatically cheap.

### 3.2 Payloads: lazy loading

A **payload** is a reference that the stage can choose not to load. Open a stage with `Usd.Stage.Open(path, Usd.Stage.LoadNone)` and every payload stays unloaded. Its prims don't exist, its files are never read, and it costs nothing. Then `stage.Load("/World/robot_3")` brings in only what you need. For robotics this matters in three places: huge environments (load the room you're in), asset features you don't always need (grippers, ROS graphs, engine-specific physics), and fast "does this file even open" checks in CI.

### 3.3 Variants: switchable alternatives in one asset

A **variant set** is a named choice with several alternatives. Only the selected variant's opinions compose. Robot assets use them for grippers, mesh fidelity, and per-engine physics. Here is the actual top level of the 6.1 Franka (`Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda`), trimmed:

```usda
#usda 1.0
( defaultPrim = "panda"  metersPerUnit = 1  upAxis = "Z" )

def Xform "panda" (
    prepend references = @./payloads/base.usda@
    variants = { string Gripper = "default"  string Mesh = "performance"  string Physics = "physx" }
    append variantSets = ["Gripper", "Mesh", "Physics"]
)
{
    variantSet "Gripper" = {
        "alternatefinger" ( prepend payload = @./payloads/Gripper/alternatefinger.usda@ ) { }
        "default"         ( prepend payload = @./payloads/Gripper/default.usda@ ) { }
        "none"            ( prepend payload = @./payloads/Gripper/none.usda@ ) { }
        "robotiq_2f_85"   ( prepend payload = @payloads/Gripper/robotiq_2f_85.usda@ ) { }
    }
    variantSet "Mesh"    = { "performance" (...) { }  "quality" (...) { } }
    variantSet "Physics" = { "none" { }  "physics" (...) { }  "physx" (...) { } }
}
```

Each variant pulls in a **payload**. Select `Gripper = robotiq_2f_85` and the Robotiq files load, while the other three grippers are never read. `Physics = none` drops the arm's physics layer and leaves an essentially visual robot, which is useful for a rendering-only twin. You select variants when you reference the asset. The official 6.1 examples pass `variants=[("Gripper", "alternatefinger"), ("Mesh", "performance")]` to `stage_utils.add_reference_to_stage`. The selection is written in **your** layer, so by LIVERPS it beats the asset's default selection.

### 3.4 Instancing: shared prototypes

Mark a prim `instanceable = true` and USD may share one **prototype** for every instanceable prim that has identical composition (the same referenced file, the same variants). The descendants of an instance are read-only proxies. You can't override a property below the instance root. In return, the renderer and other consumers can store the shared subtree once.

Isaac Sim's asset rules follow from that read-only constraint. Instance only the **mesh** subtrees, because link transforms change per environment and mesh data doesn't. Give every instanced mesh a **parent Xform** that carries the reference and the `instanceable` flag, since a mesh can't be the instance root itself. Put material bindings, physics materials and collision filters on that parent Xform, not on the mesh inside the instance.

The 6.1 Franka follows this exactly. In `payloads/base.usda` each link has a `geometry` Xform, and below it an Xform marked `instanceable = true` that references `instances.usda`. The URDF/MJCF importers were rewritten in 6.0 and produce instanceable output by default (Lecture 06), so assets you import yourself follow the same pattern.

### 3.5 How NVIDIA lays out a robot asset (Asset Structure 3.0)

The 6.x importers write one folder per robot, each file with a single job. Learn this layout, because it tells you which file to edit:

| File | Holds | Edit it when |
|---|---|---|
| `geometries.usd` | Mesh data only (binary) | CAD changed |
| `instances.usda` | Meshes + materials + collider approximations, assembled for instancing | Changing the collider choice (e.g. convex hull vs mesh) |
| `base.usda` | Kinematic hierarchy; instanceable references into `instances.usda` | Restructuring the hierarchy |
| `physics.usda` | Engine-neutral USD Physics: bodies, masses, joints, articulation root | Dynamics for all engines |
| `physx.usda` / `mujoco.usda` | Engine-specific tuning, sublayered on `physics.usda` | Per-engine tuning only |
| `robot.usda` / `materials.usda` | Isaac robot schema (`IsaacRobotAPI`) / visual materials | Robot metadata / looks only |
| top-level `<robot>.usda` (interface) | References `base.usda`, adds features as variant-selected payloads | Choosing what's switchable |

The rule NVIDIA states: source assets stay unchanged so you can re-import without losing downstream edits. Your edits go in feature layers or in the scene that uses the asset.

---

## 4. Schemas: typed vs applied

A **schema** gives a prim meaning. There are two kinds, and robot assets use both all the time:

| Kind | How it attaches | Examples | A prim has |
|---|---|---|---|
| **Typed (IsA)** | The prim's type name | `UsdGeom.Xform`, `UsdGeom.Mesh`, `UsdGeom.Cube`, `UsdPhysics.Scene`, `UsdPhysics.RevoluteJoint` | Exactly one |
| **Applied API** | Listed in `apiSchemas` metadata | `UsdPhysics.RigidBodyAPI`, `CollisionAPI`, `MassAPI`, `ArticulationRootAPI`, `PhysxSchema.PhysxRigidBodyAPI`, `PhysxSchema.PhysxSceneAPI` | Any number |
| **Multiple-apply API** | Applied with an instance name | `PhysicsDriveAPI:angular`, `PhysicsJointStateAPI:angular` | One per instance name |

How to read a robot through its schemas:

* The prim with **`PhysicsArticulationRootAPI`** is the articulation root. `Articulation("/World/robot")` resolves to it (the wrapper searches for that API under the path you give).
* **`PhysicsRigidBodyAPI`** + **`PhysicsMassAPI`** on a prim means a link (mass, inertia and center of mass live on `MassAPI`). **`PhysicsCollisionAPI`** on a geometry prim means it collides.
* **Joint prims** are typed (`PhysicsRevoluteJoint`, `PhysicsPrismaticJoint`, `PhysicsFixedJoint`) and point at two bodies through `physics:body0` / `physics:body1` relationships. Drives are the multiple-apply `PhysicsDriveAPI:<axis>`.
* **`Physx*` APIs** (from `PhysxSchema`) add PhysX-only knobs on top of standard USD Physics. In the 6.x asset layout they live in `physx.usda`, behind the `Physics` variant, so engine-specific tuning never leaks into the neutral `physics.usda`.

Revisit the raw physics scene from Lecture 01 §2.2. The 6.1 Core API docs set the PhysX scene options through the applied API:

```python
from pxr import Gf, PhysxSchema, UsdPhysics
scene = UsdPhysics.Scene.Define(stage, "/World/physics")                 # typed schema
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.8)
physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())                  # applied API schema
physx.CreateEnableCCDAttr(True)              # CCD must ALSO be enabled per body (Lecture 04)
physx.CreateEnableStabilizationAttr(True)
physx.CreateEnableGPUDynamicsAttr(False)     # CPU dynamics...
physx.CreateBroadphaseTypeAttr("MBP")        # ...with the CPU broadphase
physx.CreateSolverTypeAttr("TGS")
```

The 6.1 wrapper for the same prim is `isaacsim.core.simulation_manager.PhysxScene`. `PhysxScene("/World/physics")` creates the scene if needed, applies `PhysxSceneAPI` (and `NewtonSceneAPI`), and exposes `set_solver_type`, `set_enabled_ccd`, `set_broadphase_type`, `set_enabled_gpu_dynamics` and `set_dt`. Use the wrapper in scripts. Use raw `PhysxSchema` when you need an attribute the wrapper doesn't expose (Lecture 04).

---

## 5. Units, up-axis, and xform ops: the classic import bugs

**Stage metrics.** `metersPerUnit` and `upAxis` are layer metadata. `kilogramsPerUnit` is USD Physics metadata. Isaac Sim expects meters, kilograms, seconds and **Z-up**. CAD and DCC tools often export centimeters and Y-up. `stage_utils.get_stage_units()` / `get_stage_up_axis()` read the open stage, and `UsdGeom.GetStageMetersPerUnit(stage)` does the same in raw `pxr`. Isaac Sim enables the **Metrics Assembler**, which converts divergent distance units, mass units and up-axis when you reference a file, and `add_reference_to_stage` checks for divergent units. That conversion is a correction layered on top of the asset, not a fix to it. Check units yourself before you trust the physics. A robot that is 100× too large has 10,000× the inertia per unit mass that you expect, and its gains will be wrong.

**Scales hidden in xform ops.** Units can be "fixed" with a scale instead of metadata. The 6.1 Franka's instanced mesh references carry `xformOp:scale = (0.01, 0.01, 0.01)`: the meshes were authored in centimeters and scaled into a meter-unit stage. That is legitimate, but it means **mesh vertex values are not meters**. Any tool that reads raw points (collision generators, bounding boxes, your own scripts) must apply the full transform.

**Xform op order.** A prim's local transform is the product of the ops listed in `xformOpOrder`, applied in order. Assets in the wild mix `translate`, `orient`, `rotateXYZ`, `rotateZYX`, `scale` and `transform`; the Franka itself uses `rotateZYX` in places. The experimental wrappers normalize this: `reset_xform_op_properties=True` (the default for `Cube` and other shapes) rewrites the ops to exactly `translate`, `orient`, `scale` (double precision) while preserving the world pose.

**Rotation conventions you will cross:**

| Layer | Quaternion order | Angles |
|---|---|---|
| USD (`Gf.Quat*`, `xformOp:orient`) | w first (real, then imaginary) | Degrees (joint limits, `rotateXYZ`) |
| Isaac Sim Core (classic and experimental) | **wxyz** | Radians |
| PhysX / physics tensor views | **xyzw** | Radians |
| Isaac Lab 3.0 | **xyzw** | Radians |

The experimental `RigidPrim` reorders the PhysX tensor's xyzw into wxyz for you (you can see the index shuffle in `rigid_prim.py`). Isaac Lab 3.0 does not, and uses xyzw throughout. Any code that passes orientations between the two must convert.

---

## 6. The data pathways: USD vs Fabric vs physics tensors

| Pathway | What it is | Use it for |
|---|---|---|
| **USD** | The composed stage; CPU, per-prim API | Authoring and persistence, before and after play |
| **Fabric / USDRT** | GPU-friendly runtime store, filled from USD at load and updated by the engine every frame (world transforms, e.g. `omni:fabric:worldMatrix`) | Runtime transforms; the bridge to the renderer |
| **Physics tensors** | Batched engine state through `SimulationView`, `ArticulationView`, `RigidBodyView`; valid only after play; NumPy, torch or Warp frontends | Control loops, RL, contact forces |

NVIDIA's guidance for 6.1 in one line: **USD for authoring and persistence, Fabric for runtime transforms, tensors for batch state read/write.** The experimental wrappers sit on all three, and you pick a pathway with a backend:

```python
import isaacsim.core.experimental.utils.backend as backend_utils
with backend_utils.use_backend("fabric"):           # "usd" | "usdrt" | "fabric" | "tensor"
    positions, orientations = prims.get_world_poses()
```

| Wrapper method | Backends it supports (first = default) |
|---|---|
| `XformPrim.get_world_poses` | `usd`, `usdrt`, `fabric` |
| `RigidPrim.get_world_poses`, `Articulation.get_world_poses` | `tensor`, `usd`, `usdrt`, `fabric` |
| `RigidPrim.get_velocities`, `get_masses` | `tensor`, `usd` |
| `Articulation.get_dof_positions`, `get_jacobian_matrices` | `tensor` only |

**Why `usd` is slow in a loop.** Look at the 6.1 implementation of `XformPrim.get_world_poses` on the `usd` backend. It is a Python `for` loop that calls `ComputeLocalToWorldTransform` on each prim, one by one, then copies to the output device. The cost is CPU work per prim, plus Python overhead per prim, every call. The `fabric` backend launches one Warp kernel over all selected prims. The `tensor` backend asks the physics engine for one batched buffer. All three return the same answer when the data is fresh; at 1,024 prims they are not the same speed. Lab 2c measures by how much.

**Why `usd` can be stale.** When you put PhysX on the GPU (`SimulationManager.set_device("cuda:0")`), the 6.1 source enables `omni.physx.fabric` and sets `/physics/updateToUsd` to false: physics results go to Fabric, not back to USD. Reading poses through USD after stepping then returns the authored (or last written) values, not the simulated ones. That is correct behavior, not a bug, and it's one more reason the `usd` backend belongs to authoring time.

**Fallbacks.** If you request a backend a method doesn't support, the wrapper logs a warning and uses its first supported one. If you ask for `tensor` before play, it falls back to `usd`. During benchmarks, wrap with `use_backend(..., raise_on_unsupported=True, raise_on_fallback=True)` so a silent fallback can't fake your numbers. The `usdrt` and `fabric` backends need the **Fabric Scene Delegate**. The 6.1 base experience enables it (`app.useFabricSceneDelegate = true` in `isaacsim.exp.base.kit`), and `use_backend` raises `RuntimeError` if your app turned it off.

---

## 7. Hosted assets

`isaacsim.storage.native.get_assets_root_path()` returns the asset root. In 6.1 the default is NVIDIA's HTTPS asset service under a **versioned** path (`.../Assets/Isaac/6.1`), so asset paths are tied to the Isaac Sim release. Robots moved to `Isaac/Robots_Multiphysics/` in 6.1, so older `Isaac/Robots/...` paths may 404. Override the root with `ISAACSIM_ASSET_ROOT` (a mirror or a local folder) or `ISAACSIM_ASSET_REGION_PROFILE` (`us`, `china`), or use the offline asset packs on air-gapped machines. No Nucleus server is needed. Network latency shows up as load time, so time one open from the hosted service and one from a local copy to separate network from composition.

---

## 8. The hardware view: what USD choices cost

| USD choice | What it moves | Why |
|---|---|---|
| **Unique vs instanced meshes** | VRAM (render geometry), host RAM, load time | Instanced prims with identical composition share one prototype. Unique copies are stored separately. |
| **Payloads unloaded** | Load time, host RAM, VRAM | Unloaded prims don't exist: no composition, no geometry upload, no physics parsing. |
| **Mesh fidelity variant** (`performance` vs `quality`) | VRAM, render time, sometimes collision cost | Fewer triangles to store and trace. |
| **Collider approximation** (in `instances.usda`) | Physics ms/step, physics GPU buffers | Primitive < convex hull < convex decomposition < triangle mesh in cost (Lecture 04). |
| **Backend for runtime reads** | CPU ms/step, CPU↔GPU copies | `usd` = per-prim Python loop; `fabric` / `tensor` = one batched operation. |

A rough memory model for N copies of an asset with unique mesh data \( S_{\text{mesh}} \) and per-copy overhead \( S_{\text{copy}} \) (transforms, physics bodies, Fabric entries):

$$
\text{VRAM}_{\text{unique}} \approx N \cdot (S_{\text{mesh}} + S_{\text{copy}}), \qquad
\text{VRAM}_{\text{instanced}} \approx S_{\text{mesh}} + N \cdot S_{\text{copy}}
$$

Instancing removes the \( N \cdot S_{\text{mesh}} \) term but not the \( N \cdot S_{\text{copy}} \) term. Expect a large saving when meshes are heavy and copies are many, and little when meshes are light. Physics cost doesn't change at all: 64 instanced robots are still 64 articulations to simulate. Lab 2b measures both terms on your card.

> **8 GB budget.** On the 4060, use instanced assets for anything you clone (the importer default does this) and the `performance` mesh variant where it exists. Keep heavy environments behind payloads and load only what the task touches. Never read poses through `usd` in a stepping loop. A 1,024-copy work cell with simple meshes is a reasonable stress test on 8 GB. If unique high-fidelity meshes are what you actually need (SDG of photoreal scenes, digital twins of a full line), that's a 48 GB-class job: move it to a cloud L40S or RTX PRO 6000 rather than fighting the ceiling.

---

## 9. Build it

Everything goes in `usd_lab/`. Run every script with the Isaac Sim Python. Each one starts `SimulationApp` first, which gives `pxr` the Omniverse resolver, so `https://` asset paths open directly.

### Lab 2a — Take apart a hosted robot asset

```python
# usd_lab/inspect_asset.py
import collections, sys, time
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom, UsdPhysics
from isaacsim.storage.native import get_assets_root_path

url = get_assets_root_path() + (sys.argv[1] if len(sys.argv) > 1 else
      "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda")
for load in (Usd.Stage.LoadNone, Usd.Stage.LoadAll):            # payloads off, then on
    t = time.perf_counter(); stage = Usd.Stage.Open(url, load)
    n = sum(1 for _ in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    print(f"{load}: open {time.perf_counter() - t:.2f}s, {n} prims")
print("m/unit", UsdGeom.GetStageMetersPerUnit(stage), "kg/unit", UsdPhysics.GetStageKilogramsPerUnit(stage),
      "up", UsdGeom.GetStageUpAxis(stage))
root, vsets = stage.GetDefaultPrim(), stage.GetDefaultPrim().GetVariantSets()
print({v: (vsets.GetVariantSet(v).GetVariantNames(), vsets.GetVariantSelection(v)) for v in vsets.GetNames()})
schemas, instances = collections.Counter(), 0
for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
    schemas.update(prim.GetAppliedSchemas()); instances += prim.IsInstance()
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print("articulation root:", prim.GetPath())
    if prim.IsA(UsdPhysics.RevoluteJoint):
        j = UsdPhysics.RevoluteJoint(prim)
        print(f"  {prim.GetName()}: axis={j.GetAxisAttr().Get()} "
              f"limits=[{j.GetLowerLimitAttr().Get()}, {j.GetUpperLimitAttr().Get()}] deg")
print("applied schemas:", dict(schemas.most_common(10)))
print(f"instances={instances} prototypes={len(stage.GetPrototypes())}")
simulation_app.close()
```

Run it on the Franka, then on an arm and a mobile base of your choice from the hosted library. For each, record units, up-axis, articulation root, joint count and limits, instances and prototypes, and the prim count with payloads off and on (the difference is what the selected variants' payloads pull in).

### Lab 2b — A work cell: references, a variant set, instancing

One script writes a heavy part (a dense mesh under a parent Xform, as §3.4 requires), then a cell that references the Franka with **your** variant selections, adds a `fixture` variant set of your own, and places N copies of the part, instanced or not:

```python
# usd_lab/make_cell.py   usage: make_cell.py N INSTANCED(0|1)
import os, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import numpy as np
from pxr import Gf, Usd, UsdGeom
from isaacsim.storage.native import get_assets_root_path
n, inst = int(sys.argv[1]), bool(int(sys.argv[2]))

def new_stage(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s = Usd.Stage.CreateNew(path); UsdGeom.SetStageMetersPerUnit(s, 1.0); UsdGeom.SetStageUpAxis(s, "Z")
    return s

if not os.path.exists("usd_lab/parts/widget.usda"):              # ~37k quads; raise k for a heavier part
    s, k = new_stage("usd_lab/parts/widget.usda"), 192
    s.SetDefaultPrim(UsdGeom.Xform.Define(s, "/Widget").GetPrim())
    th, ph = np.meshgrid(np.linspace(0, np.pi, k + 1), np.linspace(0, 2 * np.pi, k + 1), indexing="ij")
    pts = 0.03 * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)
    a = (np.arange(k)[:, None] * (k + 1) + np.arange(k)[None, :]).ravel()
    mesh = UsdGeom.Mesh.Define(s, "/Widget/geom")
    mesh.CreatePointsAttr(pts.tolist()); mesh.CreateFaceVertexCountsAttr([4] * k * k)
    mesh.CreateFaceVertexIndicesAttr(np.stack([a, a + k + 1, a + k + 2, a + 1], -1).ravel().tolist()); s.Save()

s = new_stage(f"usd_lab/cell_{n}_{'inst' if inst else 'uniq'}.usda")
s.SetDefaultPrim(UsdGeom.Xform.Define(s, "/World").GetPrim())
robot = s.DefinePrim("/World/robot", "Xform")                    # reference + variant selections (local = strong)
robot.GetReferences().AddReference(get_assets_root_path() + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda")
robot.GetVariantSets().GetVariantSet("Gripper").SetVariantSelection("none")
fixture = s.DefinePrim("/World/fixture", "Xform")                # our own variant set
vset = fixture.GetVariantSets().AddVariantSet("fixture")
for name, h in (("tray", 0.02), ("bin", 0.2)):
    vset.AddVariant(name); vset.SetVariantSelection(name)
    with vset.GetVariantEditContext():
        box = UsdGeom.Cube.Define(s, "/World/fixture/body")
        box.AddTranslateOp().Set(Gf.Vec3d(0.5, 0, h / 2)); box.AddScaleOp().Set(Gf.Vec3f(0.2, 0.15, h / 2))
vset.SetVariantSelection("tray")
side = int(np.ceil(np.sqrt(n)))
for i in range(n):                                               # the instancing experiment
    w = UsdGeom.Xform.Define(s, f"/World/widgets/w_{i}")
    w.AddTranslateOp().Set(Gf.Vec3d(1.0 + 0.08 * (i % side), -0.6 + 0.08 * (i // side), 0.05))
    w.GetPrim().GetReferences().AddReference("./parts/widget.usda")
    w.GetPrim().SetInstanceable(inst)
s.Save(); app.close()
```

Then write `usd_lab/load_cell.py`: start `SimulationApp({"headless": True})`, record `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`, open the cell with `ok, stage = stage_utils.open_stage(path)`, call `simulation_app.update()` while `stage_utils.is_stage_loading()`, and stop the open-time clock. Then render 60 more frames so geometry is actually uploaded, and print open time, `len(stage.GetPrototypes())` and the VRAM delta.

Sweep `N ∈ {16, 64, 256, 1024}`, instanced and unique, one fresh process per load. Then open a cell in the GUI. Switch `fixture` to `bin` (Property panel, Variants) and watch only `/World/fixture/body` change. Try to edit `/World/widgets/w_0/geom`: in the instanced cell USD refuses, because of the read-only proxy rule from §3.4.

### Lab 2c — 1,024 pose reads through each backend

```python
# usd_lab/backend_timing.py   usage: backend_timing.py N DEVICE(cpu|cuda:0)
import sys, time
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
import numpy as np, omni.timeline, warp as wp
import isaacsim.core.experimental.utils.backend as backend_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
n, device, reps = int(sys.argv[1]), sys.argv[2], 50

stage_utils.create_new_stage()
SimulationManager.setup_simulation(dt=1.0 / 60.0, device=device)
GroundPlane("/World/ground")                                     # includes a collision plane
side, i = int(np.ceil(np.sqrt(n))), np.arange(n)
shape = Cube([f"/World/c_{k}" for k in range(n)], sizes=0.1,
             positions=np.stack([(i % side) * 0.15, (i // side) * 0.15, np.full(n, 0.5)], 1))
GeomPrim(shape.paths, apply_collision_apis=True)
rigid, xform = RigidPrim("/World/c_.*"), XformPrim("/World/c_.*")
omni.timeline.get_timeline_interface().play(); simulation_app.update()
SimulationManager.step(steps=120)                                # cubes fall and settle at z ≈ 0.05

for wrapper, backend in ((xform, "usd"), (xform, "usdrt"), (xform, "fabric"), (rigid, "tensor")):
    with backend_utils.use_backend(backend, raise_on_unsupported=True, raise_on_fallback=True):
        wrapper.get_world_poses()                                # warm-up
        t = time.perf_counter()
        for _ in range(reps):
            p, _ = wrapper.get_world_poses()
        wp.synchronize()                                         # fabric/tensor reads launch async GPU work
        ms = (time.perf_counter() - t) / reps * 1e3
    print(f"{device} n={n} {backend:7s} {ms:8.3f} ms/call  mean z={p.numpy()[:, 2].mean():.3f}")
simulation_app.close()
```

Run it on `cpu` and `cuda:0`, for `n` of 64, 1,024 and 4,096. **Speed:** plot ms/call against `n` per backend. Expect `usd` to grow roughly linearly with a large per-prim constant, and `fabric` and `tensor` to stay much flatter. **Freshness:** a mean z near 0.5 means stale data. Note which backends are stale on which device, and explain why with §6. `SimulationManager.step()` updates Fabric only with `update_fabric=True`, so also try one `simulation_app.update()` before the reads.

---

## 10. Use it in the real stack

* **Isaac Lab cloning** (Lecture 09) is USD composition at scale: one environment authored under `{ENV_REGEX_NS}`, then cloned by reference with instanced meshes. The memory model in §8 is why it fits thousands of robots on one GPU.
* **Importers** (Lecture 06) write the Asset Structure 3.0 layout from §3.5. When you tune a robot, the question is always "which layer should hold this opinion?"
* **Isaac Sim's own wrappers** read and write the schemas you inspected. Before play, `RigidPrim(..., masses=...)` authors `MassAPI`, `Articulation` finds `ArticulationRootAPI`, and drive gains land on `PhysicsDriveAPI:<axis>`. When a wrapper surprises you, open the stage and look at the attribute. Keep scene and feature layers as `.usda`, so a physics change is reviewable as a text diff.

---

## 11. Measure it

| Metric | How | Why it matters |
|---|---|---|
| **Open time, payloads off vs on** | Lab 2a (`LoadNone` vs `LoadAll`) | What lazy loading saves you |
| **VRAM delta and prototype count, instanced vs unique, vs N** | Lab 2b, `nvidia-smi`, `GetPrototypes()` | Fit \( S_{\text{mesh}} \) and \( S_{\text{copy}} \) from §8; 0 prototypes means instancing didn't happen |
| **Pose-read ms/call per backend** | Lab 2c | The cost of the wrong data path, per step |
| **Staleness per backend and device** | Lab 2c mean z | Which reads are safe after GPU physics steps |
| **Asset metrics** | Lab 2a: units, up-axis, joint limits, mass on links | The import-validation checklist used in Lecture 06 |

---

## 12. Ship it

Commit `usd_lab/` with:

* `inspect_asset.py` and `ASSETS.md`: a table of the three robots you inspected (units, up-axis, articulation root, DOF count, instances/prototypes, prim count with payloads off and on)
* `make_cell.py`, `load_cell.py`, and the generated `cell_*.usda` files (regenerate `parts/` rather than committing large meshes)
* `instancing.csv` and `instancing.png`: VRAM delta and open time vs N, instanced vs unique, with fitted \( S_{\text{mesh}} \) and \( S_{\text{copy}} \)
* `backend_timing.py`, `backends.csv` and `backends.png`: ms/call vs n per backend, on CPU and GPU physics, with a staleness column
* `USD_NOTES.md`: one paragraph on which backend your training loop will use and why, and one on where in the Franka's layer stack you would put a new joint-gain override

---

## Exit criteria

You can move on when you can:

* read a robot's `.usda` and point to its articulation root, links, joints, drives and colliders
* predict the winning value of an attribute set in an asset, a sublayer, a variant and your scene, and verify it with `GetPropertyStack()`
* explain what references, payloads, variants and instancing each cost or save, with your own measurements
* find and fix a unit or up-axis problem in an imported asset
* state the per-call cost of reading 1,024 poses through `usd`, `fabric` and `tensor` on your card, and which of them are fresh after GPU physics steps

---

## Self-check

1. You reference the hosted Franka into your scene and set `physics:upperLimit` on `panda_joint4` by editing a copy of `physics.usda`, but the joint still stops at the old limit. Name two places a stronger opinion could come from, and the call that would show you which one wins.
2. A teammate clones 512 robots for RL and finds VRAM grows almost linearly with robot count, steeply. `len(stage.GetPrototypes())` returns 0. What is wrong with the asset, and what do you expect to change after fixing it? What won't change?
3. A URDF-converted gripper falls through the table and its fingers look 100× too large. List, in order, the three things you would check in the USD before touching any physics parameter.
4. Your control loop reads 1,024 poses with `XformPrim.get_world_poses()` every step and is CPU-bound. Which backend is it using, and why is that slow? After you move PhysX to `cuda:0`, the same reads report every cube at its spawn height while the viewport shows them on the floor. Explain that too, and say what you would switch to.
5. You need the same arm with three different grippers across experiments, and you don't want to load gripper meshes you aren't using. Which composition arcs do you combine, and where does the selection get authored?

---

## References

* OpenUSD — [introduction](https://openusd.org/release/intro.html), [glossary (LIVERPS strength ordering, payloads, layers)](https://openusd.org/release/glossary.html), [scenegraph instancing](https://openusd.org/release/api/_usd__page__scenegraph_instancing.html), [UsdPhysics schemas](https://openusd.org/release/api/usd_physics_page_front.html), [transformations and xform ops](https://openusd.org/release/tut_xforms.html)
* Isaac Sim 6.1 — [OpenUSD fundamentals (layers, edit targets, units)](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/omniverse_usd/open_usd.html)
* Isaac Sim 6.1 — [Asset Structure](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/robot_setup/asset_structure.html)
* Isaac Sim 6.1 — [Instanceable assets](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/isaac_lab_tutorials/tutorial_instanceable_assets.html)
* Isaac Sim 6.1 — [Conventions (units, quaternion order, axes)](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/reference_material/reference_conventions.html)
* Isaac Sim 6.1 — [Physics data flow: USD, Fabric, tensors](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/new_physics_engine.html)
* Isaac Sim 6.1 — [Core API overview (raw USD physics scene)](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/python_scripting/core_api_overview.html)
* Isaac Sim 6.1 — [Performance optimization handbook (instancing, Fabric output, colliders)](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/reference_material/sim_performance_optimization_handbook.html)
* Isaac Sim 6.1 — [Accessing assets](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/accessing_assets.html)
* `isaacsim.core.experimental.utils` source at v6.1.0 — [`backend.py` and `stage.py`](https://github.com/isaac-sim/IsaacSim/tree/v6.1.0/source/extensions/isaacsim.core.experimental.utils/python/impl)

---

## Next in this special course

* Next: [Lecture 03 — The Core Experimental API](Lecture-03.md)
* Previous: [Lecture 01 — The Isaac Stack and Your First Simulation](Lecture-01.md)
* Back: [Isaac Sim and Isaac Lab — Overview](README.md)
