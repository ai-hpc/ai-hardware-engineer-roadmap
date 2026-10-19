# Lab 1 — Worked example: product requirements in Yocto terms

**Companion:** [Lecture 3 — Module 1](Lecture-03.md) · **Course:** [Yocto guide](../Guide.md)

This is a **complete reference answer** for Lab 1: one plausible embedded product described in **Yocto vocabulary** (MACHINE, layers, image contents, OTA, partitions). Adapt the hardware and layer names to your real BSP and Yocto release.

**Note:** Exact `MACHINE` strings, layer names, and recipe names depend on your **vendor BSP** and **Yocto branch**. Always confirm against the BSP README and the release you pinned. For deep Jetson + Yocto production practice on this roadmap, see **Phase 4 Track B** (*Orin-Nano-Yocto-BSP-Production*).

**Current support note, reviewed 2026-10-18:** NVIDIA introduced official Jetson Yocto support with JetPack 7.2, including validated recipes/reference images and contributions to OE4T. Follow the [NVIDIA support announcement](https://developer.nvidia.com/blog/deploy-agentic-ready-ai-at-the-edge-with-memory-efficiency-in-nvidia-jetpack-7-2/) and its linked BSP release instructions for a board-specific combination. This does not make the course's Scarthgap/QEMU teaching layer a validated Jetson image; the board's matching layer series and release manifest still decide compatibility.

---

## 1. Target hardware

**Product concept:** Edge AI camera node (video ingest + on-device inference).

| Item | Choice |
|------|--------|
| SoC | NVIDIA Jetson Orin Nano (example) |
| Architecture | AArch64 |
| Yocto | Vendor BSP (e.g. **meta-tegra** / NVIDIA-oriented layers) |
| Acceleration | GPU, Tensor cores (CUDA / TensorRT via vendor stacks where licensed and supported) |

**Yocto mapping**

- Set **`MACHINE`** to the value defined by your BSP (example placeholder: `jetson-orin-nano-devkit` — **verify in BSP docs**).
- Add the **BSP layer** (and its documented dependencies) to **`bblayers.conf`**.
- Kernel, firmware, and many userspace GPU pieces are **owned by vendor recipes** in that layer or companion layers.

---

## 2. Connectivity

**Required / desired interfaces**

- Ethernet (primary backhaul)
- Wi-Fi and BLE (optional product goal)
- USB host (peripherals)
- MIPI CSI-2 (camera)
- UART (serial console)

**Yocto mapping**

- **Kernel + device tree:** enable or carry the right drivers and DT nodes for your carrier and modules.
- **Userspace packages (examples):** `wpa-supplicant` or `iwd` for Wi-Fi, `bluez5` for Bluetooth, `iproute2` for networking; exact recipe names vary by layer.
- **Debug:** getty on UART is typically image / distro policy, not something you “apt install” after shipping.

---

## 3. Storage layout

**Assumption:** eMMC (or NVMe) for production; SD possible for bring-up.

**Intent**

- Boot artifacts (bootloader, kernel, DTB as required by platform)
- Root filesystem with **A/B** slots for OTA
- Writable **data** partition (logs, models, config)

**Yocto mapping**

- **`wic`** (`.wks`) or BSP-specific image types define partition tables.
- Your **image recipe** + **OTA layer** must agree on which partition is active, where the rootfs is copied, and how the bootloader selects A vs B.

Illustrative **concept only** (not a drop-in `.wks` for any specific BSP):

```
# Pseudocode — replace with BSP-correct sources and sizes
part /boot   --source bootimg-partition --size 256M
part /       --source rootfs --label rootfs_a
part /       --source rootfs --label rootfs_b
part /data   --size 1024M --fstype ext4
```

---

## 4. Update strategy

**Choice:** **Image-based OTA** with **A/B** rootfs and rollback.

**Typical stacks**

- **RAUC** (`meta-rauc` + integration work), or
- **Mender** (`meta-mender` + integration work)

**Yocto mapping**

- Add the OTA **meta-layer** and follow its **MACHINE** / **distro** integration guide.
- Bootloader (often U-Boot or platform-specific firmware) must support **switching** boot paths and failure policies you define.
- CI should produce **signed** or otherwise protected bundles if your threat model requires it (out of scope here, but plan for it).

---

## 5. Must-have userspace

**Core**

- **Init:** `systemd` (if that is your distro policy)
- **Remote access:** `openssh`
- **Logs:** journald (with `systemd`)

**Application / AI stack (product-dependent)**

- CUDA / TensorRT (usually **vendor recipes**; licensing and redistribution rules apply)
- **Python 3**
- **OpenCV** (recipe name may be `opencv`, `opencv-python`, or split packages — check your layers)
- **GStreamer** and plugins for camera pipelines

**Optional**

- **Containers:** OCI runtime / `docker` — often heavy; many teams defer to a dedicated meta-layer or vendor bundle. Treat as **explicit** image feature, not an afterthought.

**Real time**

- **Moderate** latency: often **PREEMPT** kernel is enough.
- **Hard real time:** separate exercise (PREEMPT_RT, CPU isolation, measurement). Do not assume it “for free” from a default BSP kernel.

**Example `local.conf` fragment (illustrative)**

Use **`IMAGE_INSTALL:append`** (syntax for recent Yocto) and **only** packages that exist in **your** configured layers:

```
IMAGE_INSTALL:append = " \
    openssh \
    python3 \
    python3-pip \
"
```

Add `opencv`, GStreamer packages, OTA client, etc., **after** you confirm recipe names with `bitbake-layers` / `oe-pkgdata-util` or layer documentation.

**Kernel (concept)**

- For preemptible kernel behavior, use **config fragments** or BSP-supported mechanisms; example symbol: `CONFIG_PREEMPT=y` (exact integration depends on kernel recipe).

---

## 6. Must have vs nice to have

**Must have (production-critical)**

- Boot chain + kernel + DTB for your board
- Camera path (CSI driver + userspace capture stack you will ship)
- Minimum network path (at least Ethernet for many products)
- SSH (or another controlled admin path) if operators need it
- Inference runtime you actually ship (CUDA/TensorRT or lighter stack)
- **OTA client + A/B** if you committed to image OTA
- **systemd** units (or your init policy) for the application

**Nice to have**

- Wi-Fi / BLE if not required on day one
- Docker if you can defer orchestration
- Debug tools (`strace`, `htop`, `tcpdump`) — often a **`debug`** image variant, not the production image
- GUI / Wayland — usually omitted on headless edge cameras

---

## Mindset (what this lab is training)

- Yocto builds **images** from **declarative metadata**; the question is **what is baked in**, not what you install later on a golden device.
- **Hardware alignment** flows through **MACHINE**, **BSP layers**, **kernel / DT**, and **image recipes**.
- **Reproducibility** means pinned layers, known **DISTRO** policy, and CI that rebuilds from the same inputs.

---

## Bonus: sketch of repo-facing files

**`conf/local.conf`** (fragment)

```
MACHINE ?= "YOUR_BSP_MACHINE_NAME"
DISTRO ?= "poky"
# Parallelism, download dir, sstate — set per team policy
```

**`conf/bblayers.conf`** (illustrative layer stack)

```
# Bottom-up: core + hardware + OTA + product
# Exact paths and layer names must match your checkout
BBPATH = "${TOPDIR}"
BBFILES ?= ""
```

You will add entries such as `meta`, `meta-poky`, `meta-oe`, **BSP meta**, **meta-rauc** or **meta-mender**, then **`meta-yourproduct`** for your image and bbappends.

**Custom image** (in your product layer), conceptually:

```
inherit core-image
IMAGE_INSTALL:append = " your-runtime-packages"
```

Name the image `yourproduct-image.bb` and `bitbake yourproduct-image` when the stack is wired.

---

## How to implement (workflow)

The sections above are the **what**. This section is the **how**: a realistic order of operations from empty checkout to flashable artifact. Treat it as a **blueprint**—you will adjust branch names, layer paths, `MACHINE`, and recipe names to match **your** BSP and Yocto release.

**Reality checks**

- **Layer compatibility:** Poky, `meta-openembedded`, `meta-tegra` (OE4T), and `meta-rauc` each target specific Yocto branches. Use the **combination documented** by the BSP (e.g. OE4T manifest or README), not random `git checkout` tags mixed ad hoc.
- **Jetson + RAUC + A/B:** Boot flow and partitioning on Tegra are **BSP-specific**. Expect extra integration (bootloader, signing, slot layout) beyond “add meta-rauc”.
- **CUDA / TensorRT / Docker:** Often require vendor or community layers, license acceptance, and sometimes **not** the same image as a minimal RAUC target. Prove each package with `bitbake -e` / `bitbake <recipe>` before locking the image.

**Pipeline:** host setup → clone Poky → `oe-init-build-env` → add layers → `local.conf` → product layer → image recipe → (optional) app recipe → kernel/DT tweaks → WIC/OTA wiring → `bitbake` → deploy → vendor flash tools.

### 1. Select a supported BSP baseline and initialize the workspace

**Status:** this board-specific workflow is a design/integration exercise; it has not been built or flashed in the course review. Begin with the tested native and metadata route in [Lecture 5](Lecture-05.md) and the [supplied small package](../Labs/README.md).

For a Jetson product, choose the exact upstream/BSP manifest combination documented by the board-support maintainers. Record every resolved commit and the matching host requirements. Do not select an EOL series merely because an older tutorial used it. The [official release page](https://www.yoctoproject.org/development/releases/) and the BSP compatibility documentation answer different questions; both matter.

Use this workspace layout for the relative paths below:

```text
work/
  poky/
  build/
  meta-openembedded/
  meta-tegra/
  meta-rauc/
```

After fetching the compatible pinned repositories, initialize from `work/` using `source poky/oe-init-build-env build`. You will then be in `work/build/`, so `../meta-tegra` correctly names the sibling layer. Host package installation must follow the **selected release's** official requirements rather than a frozen package list from another Ubuntu release.

### 2. Add layers (example stack)

Clone **dependencies your BSP documents** (versions must match). Illustrative layout:

```text
work/
  poky/
  meta-openembedded/
  meta-tegra/          # OE4T — check required branch next to Poky
  meta-rauc/
```

Register layers from inside the build directory:

```bash
bitbake-layers add-layer ../meta-tegra
bitbake-layers add-layer ../meta-openembedded/meta-oe
bitbake-layers add-layer ../meta-openembedded/meta-python
bitbake-layers add-layer ../meta-rauc
```

Run `bitbake-layers show-layers` and fix **BBFILE_COLLECTIONS** / dependency errors before continuing.

### 3. Configure `conf/local.conf`

Edit `build/conf/local.conf`. **Replace** `MACHINE` and distro features with values from your BSP.

```bash
# Example only — confirm MACHINE with meta-tegra / board docs
MACHINE = "jetson-orin-nano-devkit"

DISTRO_FEATURES:append = " systemd"
VIRTUAL-RUNTIME_init_manager = "systemd"

PACKAGE_CLASSES ?= "package_rpm"

# Optional: reclaim disk during build (tradeoff: harder debug of failed workdirs)
INHERIT += "rm_work"
```

**Development image** (optional): adds debug affordances; do not ship blindly to production.

```bash
EXTRA_IMAGE_FEATURES += "debug-tweaks ssh-server-openssh"
```

### 4. Create the product layer

```bash
bitbake-layers create-layer ../meta-myproduct
bitbake-layers add-layer ../meta-myproduct
```

Typical layout:

```text
meta-myproduct/
├── conf/layer.conf
├── recipes-core/images/
├── recipes-app/
├── wic/                    # if you own custom .wks
└── recipes-kernel/linux/   # bbappends or fragments, if needed
```

### 5. Custom image recipe

File: `meta-myproduct/recipes-core/images/my-edge-image.bb`

```text
SUMMARY = "Edge AI camera root filesystem (example)"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f833b3da"

inherit core-image

# Use current override syntax for the selected supported release. List existing packages only.
IMAGE_INSTALL:append = " \
    python3 \
    openssh-sshd \
"

# Add opencv, gstreamer, docker, rauc, etc. only after bitbake resolves them.
# Example placeholders (names vary by layer):
# IMAGE_INSTALL:append = " opencv gstreamer1.0-plugins-base rauc"
```

Build:

```bash
bitbake my-edge-image
```

### 6. Ship an application with systemd (pattern)

**Recipe** `meta-myproduct/recipes-app/myapp/myapp_1.0.bb`. Put `app.py` and `myapp.service` in the same directory as the `.bb` file (or under a `files/` subdirectory, following your layer’s FILESPATH convention).

```text
SUMMARY = "Example AI app service"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f833b3da"

SRC_URI = "file://app.py \
           file://myapp.service \
          "

S = "${WORKDIR}"

RDEPENDS:${PN} += "python3"

do_install() {
    install -d ${D}${bindir}
    install -m 0755 ${WORKDIR}/app.py ${D}${bindir}/myapp.py

    install -d ${D}${systemd_system_unitdir}
    install -m 0644 ${WORKDIR}/myapp.service ${D}${systemd_system_unitdir}
}

inherit systemd

SYSTEMD_SERVICE:${PN} = "myapp.service"
```

**`files/myapp.service`**

```ini
[Unit]
Description=Example edge AI app

[Service]
ExecStart=/usr/bin/python3 /usr/bin/myapp.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Add `myapp` to `IMAGE_INSTALL:append` in the image recipe.

### 7. Kernel and hardware

- **Menuconfig (when supported):** `bitbake virtual/kernel -c menuconfig` — some BSPs prefer **only** fragments; follow vendor docs.
- **Fragments / defconfig:** prefer `*.cfg` fragments and `bbappend` to `linux-yocto` or the BSP kernel recipe instead of hand-editing trees you cannot rebase.
- **Device tree:** `.dts`/`.dtsi` in your layer; apply via `SRC_URI` + patch or via BSP extension mechanisms.

### 8. Storage layout (WIC)

Place a `.wks` under your layer (e.g. `meta-myproduct/wic/my-layout.wks`). **Do not** assume the partition stanza below works on Tegra without BSP alignment.

```bash
part /boot --source bootimg-partition --fstype=vfat --label boot --size 256M
part / --source rootfs --fstype=ext4 --label rootfs_a
part / --source rootfs --fstype=ext4 --label rootfs_b
part /data --fstype=ext4 --size 1024M
```

Point the **image** at it. Put the `.wks` under `wic/` in your layer; then in the image recipe (names vary by release):

```text
WKS_FILE = "my-layout.wks"
IMAGE_FSTYPES:append = " wic"
```

If BitBake cannot find the file, check `WKS_SEARCH_PATH` / layer `BBFILE_COLLECTIONS` and the docs for your Yocto version.

A/B + RAUC usually requires matching **bootloader env**, **RAUC system.conf**, and **partition UUIDs**—follow `meta-rauc` integration for your SoC.

### 9. OTA (RAUC outline)

- Add RAUC **distro** / **image** features per `meta-rauc` documentation for your branch.
- Install the RAUC client into the image (`rauc` package when available).
- Define **slots**, **keys**, and **bundle** recipes in CI; signing is mandatory for serious deployments.

Mender follows a parallel path via `meta-mender` with its own partition assumptions.

### 10. Build and artifacts

```bash
bitbake my-edge-image
```

Inspect deploy (path varies by `MACHINE` and `TMPDIR`):

```text
tmp/deploy/images/<MACHINE>/
  *.wic
  *.ext4 / tar / other fstypes
  kernel / dtb artifacts (BSP-dependent)
```

### 11. Flash to hardware

Use **vendor tools** for the platform (for Jetson, NVIDIA’s flashing workflow / BSP docs—not a generic one-liner). OE4T and L4T document where images land and how they map to `flash.sh` or equivalent.

### Traditional distro vs Yocto (engineering view)

| Traditional Linux | Yocto |
|-------------------|--------|
| Install packages after boot | Decide image contents at build time |
| Manual snowflake setup | Automated, reviewable metadata |
| Hard to reproduce | Reproducible with pinned layers and cache policy |

### Where to go deeper on this roadmap

- **Failed builds and logs:** [Lecture 11 — Debugging builds](Lecture-11.md)
- **Performance / sstate / CI:** [Lecture 13 — Performance, caching, and CI](Lecture-13.md)
- **bbappend discipline:** [Lecture 7 — Layers](Lecture-07.md)
- **Jetson Yocto production:** Phase 4 Track B — *Orin-Nano-Yocto-BSP-Production*

---

## Next steps on this roadmap

1. Continue [Lecture 4 — Architecture](Lecture-04.md) (layers and configs).
2. When you move to Jetson-specific BSP depth, use **Phase 4 Track B** Yocto production material alongside this spec.

---

**Previous:** [Lecture 3](Lecture-03.md) | **Next:** [Lecture 4](Lecture-04.md)
