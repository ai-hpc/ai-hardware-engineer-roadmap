# Jetson Orin Nano 8GB (T234) -- Security Architecture and Production Hardening Guide

<div class="course-identity auto-course" style="--course-accent: #65a30d; --course-accent-rgb: 101, 163, 13;" markdown="1">
<div class="course-identity__icon">JON8</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Jetson Orin Nano 8GB (T234) -- Security Architecture and Production Hardening Guide.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


This guide provides an end-to-end deep dive into the security subsystem of the
NVIDIA Jetson Orin Nano 8GB developer kit and production module. It covers
hardware-rooted trust, secure boot, disk encryption, trusted execution, network
hardening, firmware update integrity, model protection, and production
deployment checklists. All commands and paths reference JetPack 6.x / L4T 36.x
on the T234 SoC unless stated otherwise.

---

## 1. Introduction

### 1.1 Why Security Matters for Edge AI

Edge AI devices operate outside the controlled perimeter of a data center.
A Jetson Orin Nano deployed in a smart camera, robotics platform, or medical
device faces physical access threats that server-side GPUs never encounter.
Attackers can:

- Physically extract the eMMC or SD card and read the filesystem.
- Probe JTAG/SWD debug ports to dump firmware or inject code.
- Intercept UART console output to harvest credentials.
- Tamper with the boot chain to load malicious kernels.
- Steal proprietary AI models stored in plaintext on disk.
- Perform man-in-the-middle attacks on OTA update channels.

A single compromised device can become a pivot point into the broader fleet
management network, exfiltrate model intellectual property, or produce
manipulated inference results in safety-critical applications.

### 1.2 Threat Model for Deployed Jetson Devices

The following threat model should guide every architectural decision in this
guide.

```
+-----------------------------------------------------------------------+
|                        THREAT MODEL SUMMARY                           |
+-------------------+---------------------------------------------------+
| Threat Category   | Examples                                          |
+-------------------+---------------------------------------------------+
| Physical Access   | eMMC dump, JTAG probing, cold-boot attack,        |
|                   | bus snooping (SPI/I2C), board cloning              |
+-------------------+---------------------------------------------------+
| Boot Tampering    | Modified bootloader, unsigned kernel, rootkit      |
|                   | injection via recovery mode                        |
+-------------------+---------------------------------------------------+
| Runtime Exploit   | Kernel CVE exploitation, container escape,         |
|                   | privilege escalation, buffer overflow in drivers   |
+-------------------+---------------------------------------------------+
| Network Attack    | MitM on OTA channel, DNS spoofing, replay of       |
|                   | old firmware, credential theft over unencrypted    |
|                   | management interfaces                              |
+-------------------+---------------------------------------------------+
| Supply Chain      | Counterfeit modules, firmware substitution during   |
|                   | manufacturing, untrusted component insertion        |
+-------------------+---------------------------------------------------+
| IP Theft          | Model weight extraction, algorithm reverse          |
|                   | engineering, training data exfiltration             |
+-------------------+---------------------------------------------------+
| Denial of Service | Sensor spoofing, resource exhaustion, watchdog      |
|                   | timer manipulation                                  |
+-------------------+---------------------------------------------------+
```

### 1.3 Security Principles Applied Throughout This Guide

1. **Defense in Depth** -- No single mechanism is sufficient. Layer hardware
   root of trust, authenticated boot, encrypted storage, runtime confinement,
   and network controls.
2. **Least Privilege** -- Every process, container, and user gets only the
   permissions required for its function.
3. **Fail Secure** -- If verification fails at any boot stage, the device must
   halt rather than boot into an untrusted state.
4. **Attestation** -- The device must be able to prove its identity and
   integrity state to a remote verifier.

---

## 2. T234 Security Architecture

### 2.1 SoC Security Block Diagram

```
+------------------------------------------------------------------+
|                      T234 SoC (Orin Nano)                        |
|                                                                  |
|  +------------+   +------------+   +----------+   +----------+   |
|  | Cortex-A78 |   | GPU        |   | DLA      |   | VIC      |   |
|  | (6 cores)  |   | (Ampere)   |   | (x1)     |   |          |   |
|  +-----+------+   +------------+   +----------+   +----------+   |
|        |                                                         |
|  +-----v--------------------------------------------------+     |
|  |              ARM TrustZone (EL3 / Secure World)         |     |
|  |  +-------------+  +------------+  +---------------+     |     |
|  |  | OP-TEE OS   |  | Secure     |  | ATF (ARM      |     |     |
|  |  | (S-EL1)     |  | Monitor    |  | Trusted       |     |     |
|  |  |             |  | (EL3)      |  | Firmware)     |     |     |
|  |  +-------------+  +------------+  +---------------+     |     |
|  +----------------------------------------------------------+    |
|                                                                  |
|  +----------------------------------------------------------+    |
|  |              Security Engine (SE) Hardware                |    |
|  |  +--------+  +--------+  +--------+  +---------------+   |    |
|  |  | AES    |  | SHA    |  | RSA/   |  | True Random   |   |    |
|  |  | Engine |  | Engine |  | ECC    |  | Number Gen    |   |    |
|  |  |        |  |        |  | (PKA)  |  | (TRNG/DRBG)  |   |    |
|  |  +--------+  +--------+  +--------+  +---------------+   |    |
|  +----------------------------------------------------------+    |
|                                                                  |
|  +----------------------------------------------------------+    |
|  |              Fuse Controller                              |    |
|  |  +------------------+  +------------------+               |    |
|  |  | OEM Key Hash     |  | Security Mode    |               |    |
|  |  | (PKC / SBK)      |  | Fuse             |               |    |
|  |  +------------------+  +------------------+               |    |
|  |  +------------------+  +------------------+               |    |
|  |  | JTAG Disable     |  | Key Slots        |               |    |
|  |  | Fuse             |  | (KEK0/1/2)       |               |    |
|  |  +------------------+  +------------------+               |    |
|  +----------------------------------------------------------+    |
+------------------------------------------------------------------+
```

### 2.2 Security Engine (SE)

The T234 integrates a dedicated Security Engine accessible from both the secure
world and (with restrictions) from the normal world via the Linux kernel
`tegra-se` driver.

Key capabilities:

| Feature          | Details                                              |
|------------------|------------------------------------------------------|
| AES              | 128/192/256-bit, CBC/CTR/ECB/GCM modes               |
| SHA              | SHA-256, SHA-384, SHA-512                             |
| RSA              | Up to 4096-bit key length via PKA                    |
| ECC              | NIST P-256, P-384                                    |
| TRNG             | NIST SP 800-90B compliant entropy source             |
| DRBG             | CTR_DRBG per NIST SP 800-90A                         |
| Key Slots        | Hardware key slots loaded from fuses at boot          |
| DMA              | Scatter-gather DMA for bulk crypto operations         |

The SE processes crypto operations without exposing key material to the CPU.
Fuse-derived keys (SBK, KEK0, KEK1, KEK2) are loaded into SE key slots by the
BootROM and are never readable by software.

```c
/* Kernel-space access to SE -- tegra_se driver ioctl example (simplified) */
#include <linux/tegra_se.h>

struct tegra_se_aes_request req = {
    .op       = TEGRA_SE_AES_OP_ENCRYPT,
    .mode     = TEGRA_SE_AES_MODE_CBC,
    .keyslot  = TEGRA_SE_KEYSLOT_KEK2,   /* hardware-bound key */
    .keylen   = 256,
    .iv       = iv_buf,
    .src      = plaintext_buf,
    .dst      = ciphertext_buf,
    .len      = data_len,
};
ioctl(se_fd, TEGRA_SE_AES_ENCRYPT, &req);
```

### 2.3 Public Key Accelerator (PKA)

The PKA offloads RSA and ECC operations from the CPU. During secure boot the
BootROM uses the PKA to verify RSA-PSS or ECDSA signatures on boot components.
In Linux, the PKA is exposed via the kernel crypto API:

```bash
# Verify SE/PKA is registered as a kernel crypto provider
cat /proc/crypto | grep -A5 "tegra"
```

### 2.4 True Random Number Generator (TRNG)

The T234 TRNG provides hardware entropy fed into `/dev/hwrng`. The kernel
`rngd` daemon mixes this into the primary entropy pool.

```bash
# Check hardware RNG device
ls -l /dev/hwrng

# Read raw hardware entropy (test only -- do not use directly for keys)
dd if=/dev/hwrng bs=32 count=1 | xxd

# Verify entropy pool health
cat /proc/sys/kernel/random/entropy_avail
```

### 2.5 ARM TrustZone on T234

TrustZone partitions the Cortex-A78 AE cores into two worlds:

```
+----------------------------+----------------------------+
|       Secure World         |      Normal World          |
+----------------------------+----------------------------+
| EL3: ARM Trusted Firmware  |                            |
|      (ATF / BL31)          |                            |
+----------------------------+                            |
| S-EL1: OP-TEE OS           | EL1: Linux Kernel          |
+----------------------------+----------------------------+
| S-EL0: Trusted Apps (TAs)  | EL0: User-space apps       |
+----------------------------+----------------------------+
```

Memory regions are partitioned via the TrustZone Address Space Controller
(TZASC). The DRAM carve-out for OP-TEE is configured by MB2 and is invisible
to Linux:

```bash
# View TrustZone memory carve-outs from device tree
dtc -I dtb -O dts /boot/dtb/kernel_tegra234-p3767-0003-p3768-0000-a0.dtb 2>/dev/null | \
    grep -A5 "trust-zone"
```

### 2.6 Fuse-Based Key Storage Overview

The T234 contains one-time-programmable (OTP) fuses organized into security
groups:

| Fuse Name             | Width   | Purpose                                   |
|-----------------------|---------|-------------------------------------------|
| `public_key_hash`     | 512-bit | SHA-512 hash of OEM RSA/ECC public key    |
| `secure_boot_key`     | 128-bit | Symmetric key for bootloader encryption   |
| `kek0`                | 128-bit | Key encryption key slot 0                 |
| `kek1`                | 128-bit | Key encryption key slot 1                 |
| `kek2`                | 256-bit | Key encryption key slot 2 (disk encrypt)  |
| `security_mode`       | 1-bit   | Enforces secure boot (irreversible)       |
| `jtag_disable`        | 1-bit   | Disables JTAG debug (irreversible)        |
| `odm_lock`            | 4-bit   | Locks ODM fuse regions                    |
| `arm_jtag_disable`    | 1-bit   | Disables ARM debug access port            |

Once the `security_mode` fuse is burned, the BootROM will refuse to execute
any unsigned or incorrectly signed boot payload.

---

## 3. Secure Boot Chain

### 3.1 Boot Flow Overview

```
+----------+     +---------+     +---------+     +--------+     +--------+
| BootROM  | --> | MB1     | --> | MB2     | --> | UEFI   | --> | Kernel |
| (Fused   |     | (BPMP   |     | (TOS /  |     | (UEFI  |     | (Linux |
|  PKC     |     |  FW)    |     |  OP-TEE |     |  Sec   |     |  + DTB)|
|  verify) |     |         |     |  + ATF) |     |  Boot) |     |        |
+----------+     +---------+     +---------+     +--------+     +--------+
     |                |               |               |              |
     v                v               v               v              v
  OTP Fuse         RSA/ECC         RSA/ECC          db/dbx       kexec or
  Root of          Signature       Signature        Signature     dm-verity
  Trust            Check           Check            Check         Root Hash
```

### 3.2 BootROM Verification (Stage 0)

The BootROM is mask-programmed into the T234 die and cannot be modified. When
`security_mode` fuse is burned:

1. BootROM reads the `public_key_hash` fuse.
2. BootROM loads the BCT (Boot Configuration Table) from the boot device.
3. The BCT contains the OEM public key and an RSA-PSS signature.
4. BootROM hashes the embedded public key with SHA-512 and compares against
   the fuse value.
5. BootROM verifies the BCT signature using the authenticated public key.
6. BootROM decrypts MB1 using the SBK fuse (if encryption enabled).
7. BootROM verifies MB1 signature.
8. Execution transfers to MB1.

If any check fails, the BootROM halts. There is no fallback.

### 3.3 Generating Signing Keys

```bash
# Generate RSA-3072 key pair for secure boot (NVIDIA recommended minimum)
openssl genrsa -out oem_rsa3072.pem 3072

# Extract public key
openssl rsa -in oem_rsa3072.pem -pubout -out oem_rsa3072_pub.pem

# Generate the PKC hash for fuse programming
# This uses NVIDIA's tegrasign tool from the L4T BSP
cd /path/to/Linux_for_Tegra
python3 bootloader/tegrasign_v3.py \
    --pubkeyhash oem_pkc_hash.bin \
    --key oem_rsa3072.pem

# View the hash (this goes into the public_key_hash fuse)
xxd oem_pkc_hash.bin
```

For ECC P-256 signing (supported on T234):

```bash
# Generate ECC P-256 key pair
openssl ecparam -genkey -name prime256v1 -out oem_ecc_p256.pem

# Extract public key
openssl ec -in oem_ecc_p256.pem -pubout -out oem_ecc_p256_pub.pem
```

### 3.4 Signing Boot Components

The `l4t_sign_image.sh` script (or the lower-level `tegraflash.py`) signs all
boot chain components:

```bash
cd /path/to/Linux_for_Tegra

# Sign all boot components for Orin Nano 8GB (p3767-0003)
sudo ./tools/kernel_flash/l4t_initrd_flash.sh \
    --sign \
    -u oem_rsa3072.pem \
    -v pkc \
    jetson-orin-nano-devkit \
    internal

# Or using tegraflash directly for finer control:
sudo python3 bootloader/tegraflash.py \
    --chip 0x23 \
    --applet mb1_t234_prod.bin \
    --cmd "sign" \
    --key oem_rsa3072.pem \
    --encrypt_key sbk.key \
    --cfg flash_t234_qspi_sdmmc.xml
```

### 3.5 MB1 and MB2 Signature Verification

MB1 (Microboot1) runs on the BPMP (Boot and Power Management Processor).
It initializes DRAM, loads MB2, and verifies MB2's signature:

```
MB1 Verification Steps:
  1. Read MB2 binary from boot device (QSPI-NOR or eMMC)
  2. Read MB2 signature block (appended or in BCH partition)
  3. Load OEM public key from BCT (already authenticated by BootROM)
  4. Perform RSA-PSS verify(MB2_binary, signature, OEM_pubkey)
  5. If encryption enabled: AES-CBC decrypt MB2 using SBK key slot
  6. Transfer execution to MB2
```

MB2 in turn loads and verifies:
- ARM Trusted Firmware (BL31)
- OP-TEE (BL32)
- UEFI bootloader (BL33)

### 3.6 UEFI Secure Boot Configuration

JetPack 6.x uses UEFI as the primary bootloader (replacing CBoot). UEFI
Secure Boot adds another verification layer for the kernel and initrd:

```bash
# Check current UEFI Secure Boot state
mokutil --sb-state

# List enrolled keys
mokutil --list-enrolled

# Enroll a custom UEFI Secure Boot key (db key)
# 1. Generate a self-signed certificate
openssl req -new -x509 -newkey rsa:2048 -sha256 -days 3650 \
    -subj "/CN=OEM UEFI Secure Boot/" \
    -keyout uefi_db.key -out uefi_db.crt -nodes

# 2. Convert to DER for UEFI enrollment
openssl x509 -in uefi_db.crt -outform DER -out uefi_db.der

# 3. Create an EFI Signature List
cert-to-efi-sig-list uefi_db.crt uefi_db.esl

# 4. Sign the ESL with the KEK
sign-efi-sig-list -k KEK.key -c KEK.crt db uefi_db.esl uefi_db.auth
```

### 3.7 Signing the Linux Kernel for UEFI Secure Boot

```bash
# Sign the kernel image with the UEFI db key
sbsign --key uefi_db.key --cert uefi_db.crt \
    --output /boot/Image.signed /boot/Image

# Verify the signature
sbverify --cert uefi_db.crt /boot/Image.signed

# Sign kernel modules (required when module signature enforcement is on)
/usr/src/linux-headers-$(uname -r)/scripts/sign-file \
    sha256 \
    uefi_db.key \
    uefi_db.crt \
    /lib/modules/$(uname -r)/kernel/drivers/example/example.ko
```

### 3.8 Kernel Configuration for Signature Enforcement

```bash
# Verify kernel config options for module signature enforcement
zcat /proc/config.gz | grep -E "MODULE_SIG|LOCK_DOWN"

# Expected for production:
# CONFIG_MODULE_SIG=y
# CONFIG_MODULE_SIG_FORCE=y
# CONFIG_MODULE_SIG_SHA256=y
# CONFIG_LOCK_DOWN_KERNEL_FORCE_INTEGRITY=y
```

### 3.9 Complete Chain of Trust Verification

```bash
#!/bin/bash
# verify_chain_of_trust.sh -- Runtime verification of boot chain integrity

echo "=== Secure Boot Chain Verification ==="

# 1. Check if secure boot fuse is burned
FUSE_SEC=$(cat /sys/devices/platform/efuse-burn/security_mode 2>/dev/null)
echo "[1] Security mode fuse: ${FUSE_SEC:-not readable from userspace}"

# 2. Check UEFI Secure Boot status
SB_STATE=$(mokutil --sb-state 2>/dev/null || echo "mokutil not available")
echo "[2] UEFI Secure Boot: $SB_STATE"

# 3. Check kernel lockdown mode
LOCKDOWN=$(cat /sys/kernel/security/lockdown 2>/dev/null)
echo "[3] Kernel lockdown: ${LOCKDOWN:-not available}"

# 4. Check module signature enforcement
MOD_SIG=$(cat /proc/config.gz 2>/dev/null | gunzip | grep MODULE_SIG_FORCE)
echo "[4] Module signature: ${MOD_SIG}"

# 5. Check dm-verity status on rootfs
VERITY=$(dmsetup status 2>/dev/null | grep verity)
echo "[5] dm-verity: ${VERITY:-not active}"

echo "=== End Verification ==="
```

---

## 4. OP-TEE (Trusted Execution Environment)

### 4.1 OP-TEE Architecture on T234

OP-TEE runs in the Secure World (S-EL1) and provides a GlobalPlatform
TEE-compliant environment for Trusted Applications (TAs).

```
Normal World                          Secure World
+------------------+                  +------------------+
| Linux User App   |                  | Trusted App (TA) |
| (CA - Client     |                  | (S-EL0)          |
|  Application)    |                  |                  |
+--------+---------+                  +--------+---------+
         |                                     |
+--------v---------+                  +--------v---------+
| TEE Client API   |                  | OP-TEE Internal  |
| (libteec.so)     |                  | API              |
+--------+---------+                  +--------+---------+
         |                                     |
+--------v---------+                  +--------v---------+
| tee-supplicant   |                  | OP-TEE OS Core   |
| (daemon)         |                  | (S-EL1)          |
+--------+---------+                  +--------+---------+
         |                                     |
+--------v-----------------------------------------v-----+
|              ARM Trusted Firmware (EL3)                 |
|              SMC Handler / Secure Monitor               |
+---------------------------------------------------------+
```

### 4.2 OP-TEE Build and Integration

OP-TEE is built as part of the L4T BSP. On JetPack 6.x:

```bash
# OP-TEE source location in L4T BSP
ls Linux_for_Tegra/source/nvidia-ote/

# Key components:
# - optee/optee_os/       : OP-TEE OS kernel
# - optee/optee_client/   : Client library (libteec) and tee-supplicant
# - optee/optee_test/     : xtest suite for validation
# - optee/samples/hwkey-agent/ : NVIDIA hardware key agent TA

# Build OP-TEE OS for T234
cd Linux_for_Tegra/source/nvidia-ote/optee/optee_os
make \
    CROSS_COMPILE=aarch64-linux-gnu- \
    PLATFORM=tegra-t234 \
    CFG_ARM64_core=y \
    CFG_TEE_CORE_LOG_LEVEL=2 \
    CFG_REE_FS=y \
    CFG_RPMB_FS=y

# Build client library
cd ../optee_client
make \
    CROSS_COMPILE=aarch64-linux-gnu-
```

### 4.3 Verifying OP-TEE is Running

```bash
# Check OP-TEE driver is loaded
dmesg | grep -i optee

# Expected output:
# [    1.234567] optee: probing for conduit method.
# [    1.234589] optee: revision 3.18 (xxxxx)
# [    1.234601] optee: initialized driver

# Check TEE device node
ls -l /dev/tee*
# crw-rw---- 1 root tee 247, 0 ... /dev/tee0
# crw-rw---- 1 root tee 247, 16 ... /dev/teepriv0

# Check tee-supplicant is running
systemctl status tee-supplicant
# or
ps aux | grep tee-supplicant

# Run OP-TEE xtest suite (basic smoke test)
xtest -l 0
```

### 4.4 Writing a Trusted Application (TA)

A TA has two components: the TA itself (runs in secure world) and a Client
Application (CA) that invokes it from Linux.

**TA Header (ta_example.h):**

```c
#ifndef TA_EXAMPLE_H
#define TA_EXAMPLE_H

/* UUID for our example TA: 8aaaf200-2450-11e4-abe2-0002a5d5c51b */
#define TA_EXAMPLE_UUID \
    { 0x8aaaf200, 0x2450, 0x11e4, \
      { 0xab, 0xe2, 0x00, 0x02, 0xa5, 0xd5, 0xc5, 0x1b } }

/* Command IDs */
#define TA_CMD_ENCRYPT   0
#define TA_CMD_DECRYPT   1
#define TA_CMD_DERIVE    2

#endif /* TA_EXAMPLE_H */
```

**TA Implementation (ta_example.c):**

```c
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include "ta_example.h"

TEE_Result TA_CreateEntryPoint(void)
{
    DMSG("TA_CreateEntryPoint called");
    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void)
{
    DMSG("TA_DestroyEntryPoint called");
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                     TEE_Param params[4],
                                     void **session_ctx)
{
    (void)param_types;
    (void)params;
    (void)session_ctx;
    DMSG("Session opened");
    return TEE_SUCCESS;
}

void TA_CloseSessionEntryPoint(void *session_ctx)
{
    (void)session_ctx;
    DMSG("Session closed");
}

static TEE_Result aes_encrypt(uint32_t param_types, TEE_Param params[4])
{
    TEE_Result res;
    TEE_OperationHandle op = TEE_HANDLE_NULL;
    TEE_ObjectHandle key = TEE_HANDLE_NULL;

    void *in_buf = params[0].memref.buffer;
    size_t in_len = params[0].memref.size;
    void *out_buf = params[1].memref.buffer;
    size_t out_len = params[1].memref.size;

    /* Allocate a transient AES-256 key object */
    res = TEE_AllocateTransientObject(TEE_TYPE_AES, 256, &key);
    if (res != TEE_SUCCESS) return res;

    /* In production: derive key from hardware unique key (HUK) */
    /* TEE_GenerateKey or TEE_PopulateTransientObject with HUK-derived material */
    res = TEE_GenerateKey(key, 256, NULL, 0);
    if (res != TEE_SUCCESS) goto cleanup;

    res = TEE_AllocateOperation(&op, TEE_ALG_AES_CBC_NOPAD,
                                 TEE_MODE_ENCRYPT, 256);
    if (res != TEE_SUCCESS) goto cleanup;

    res = TEE_SetOperationKey(op, key);
    if (res != TEE_SUCCESS) goto cleanup;

    uint8_t iv[16] = {0}; /* In production: use random IV */
    TEE_CipherInit(op, iv, sizeof(iv));

    res = TEE_CipherDoFinal(op, in_buf, in_len, out_buf, &out_len);
    params[1].memref.size = out_len;

cleanup:
    TEE_FreeOperation(op);
    TEE_FreeTransientObject(key);
    return res;
}

TEE_Result TA_InvokeCommandEntryPoint(void *session_ctx,
                                       uint32_t cmd_id,
                                       uint32_t param_types,
                                       TEE_Param params[4])
{
    (void)session_ctx;

    switch (cmd_id) {
    case TA_CMD_ENCRYPT:
        return aes_encrypt(param_types, params);
    case TA_CMD_DECRYPT:
        /* Similar to encrypt with TEE_MODE_DECRYPT */
        return TEE_ERROR_NOT_IMPLEMENTED;
    default:
        return TEE_ERROR_BAD_PARAMETERS;
    }
}
```

### 4.5 Client Application (CA) in Linux User Space

```c
/* ca_example.c -- Client application that calls the TA */
#include <stdio.h>
#include <string.h>
#include <tee_client_api.h>
#include "ta_example.h"

int main(void)
{
    TEEC_Result res;
    TEEC_Context ctx;
    TEEC_Session sess;
    TEEC_Operation op;
    TEEC_UUID uuid = TA_EXAMPLE_UUID;
    uint32_t err_origin;

    /* Initialize TEE context */
    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TEEC_InitializeContext failed: 0x%x\n", res);
        return 1;
    }

    /* Open session with the TA */
    res = TEEC_OpenSession(&ctx, &sess, &uuid,
                           TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TEEC_OpenSession failed: 0x%x origin: 0x%x\n",
                res, err_origin);
        TEEC_FinalizeContext(&ctx);
        return 1;
    }

    /* Prepare buffers */
    char plaintext[64] = "Sensitive inference parameters for Orin Nano";
    char ciphertext[64] = {0};

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_MEMREF_TEMP_INPUT,   /* plaintext */
        TEEC_MEMREF_TEMP_OUTPUT,  /* ciphertext */
        TEEC_NONE,
        TEEC_NONE
    );
    op.params[0].tmpref.buffer = plaintext;
    op.params[0].tmpref.size = sizeof(plaintext);
    op.params[1].tmpref.buffer = ciphertext;
    op.params[1].tmpref.size = sizeof(ciphertext);

    /* Invoke TA command */
    res = TEEC_InvokeCommand(&sess, TA_CMD_ENCRYPT, &op, &err_origin);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TEEC_InvokeCommand failed: 0x%x origin: 0x%x\n",
                res, err_origin);
    } else {
        printf("Encryption successful, %zu bytes produced\n",
               op.params[1].tmpref.size);
    }

    TEEC_CloseSession(&sess);
    TEEC_FinalizeContext(&ctx);
    return (res == TEEC_SUCCESS) ? 0 : 1;
}
```

Build the CA:

```bash
# Compile client application
aarch64-linux-gnu-gcc -o ca_example ca_example.c \
    -I/usr/include \
    -lteec

# Deploy TA binary to the device
# TAs are stored in /lib/optee_armtz/ with their UUID as filename
scp 8aaaf200-2450-11e4-abe2-0002a5d5c51b.ta \
    jetson:/lib/optee_armtz/
```

### 4.6 NVIDIA Hardware Key Agent

NVIDIA ships a specialized TA called `hwkey-agent` that provides access to
hardware-unique keys derived from fuse values. This is critical for disk
encryption and secure storage:

```bash
# The hwkey-agent TA UUID
# 82154947-c1bc-4bdf-b89d-56bf45e4110b (NVIDIA hwkey-agent)

# Check if hwkey-agent is deployed
ls /lib/optee_armtz/ | grep 82154947

# The hwkey-agent derives keys using:
#   - Device-unique hardware key (HUK) from fuses
#   - Caller TA identity (UUID)
#   - Key derivation context string
# This ensures derived keys are unique per-device and per-TA
```

---

## 5. Disk Encryption

### 5.1 Encryption Strategy Overview

For production Jetson Orin Nano deployments, disk encryption protects data at
rest against physical extraction attacks. The recommended architecture uses:

- **LUKS2** (Linux Unified Key Setup) for the root filesystem partition.
- **Hardware-accelerated AES** via the T234 Security Engine.
- **OP-TEE / hwkey-agent** for key unsealing so that the disk encryption key
  is never exposed in plaintext outside the secure world.

```
Boot Flow with Encrypted Rootfs:

BootROM -> MB1 -> MB2 -> ATF + OP-TEE -> UEFI -> initrd
                                                     |
                                          +----------v-----------+
                                          | initrd mounts tmpfs  |
                                          | loads tee-supplicant |
                                          | calls hwkey-agent TA |
                                          | to unseal LUKS key   |
                                          | cryptsetup luksOpen  |
                                          | pivot_root to rootfs |
                                          +----------------------+
```

### 5.2 Creating an Encrypted Root Filesystem

```bash
# On the host machine during image preparation

# 1. Create a raw rootfs image
dd if=/dev/zero of=rootfs_encrypted.img bs=1M count=14336  # 14GB for Orin Nano

# 2. Set up LUKS2 container
# Use AES-256-XTS which maps well to T234 SE hardware acceleration
cryptsetup luksFormat \
    --type luks2 \
    --cipher aes-xts-plain64 \
    --key-size 512 \
    --hash sha256 \
    --pbkdf argon2id \
    --label jetson-rootfs \
    rootfs_encrypted.img

# 3. Open the LUKS container
cryptsetup luksOpen rootfs_encrypted.img jetson-rootfs

# 4. Create ext4 filesystem
mkfs.ext4 -L rootfs /dev/mapper/jetson-rootfs

# 5. Mount and populate with L4T rootfs
mount /dev/mapper/jetson-rootfs /mnt/rootfs
cp -a Linux_for_Tegra/rootfs/* /mnt/rootfs/

# 6. Unmount and close
umount /mnt/rootfs
cryptsetup luksClose jetson-rootfs
```

### 5.3 Hardware-Accelerated AES via dm-crypt

The T234 Security Engine registers as a kernel crypto driver. When dm-crypt
is configured, it can offload AES operations to hardware:

```bash
# On the Jetson device -- check available crypto accelerators
cat /proc/crypto | grep -B2 "driver.*tegra"

# Expected output includes:
# name         : xts(aes)
# driver       : xts-aes-tegra
# module       : tegra_se_aes
# priority     : 300       <-- higher priority than software fallback

# Verify dm-crypt is using the hardware driver
dmsetup table --showkeys jetson-rootfs
# Output: 0 <size> crypt aes-xts-plain64 <key> 0 /dev/mmcblk0p1 0

# Benchmark hardware vs software AES
cryptsetup benchmark -c aes-xts-plain64 -s 512
# On Orin Nano with SE acceleration:
#   Encrypt: ~800 MiB/s   Decrypt: ~850 MiB/s
# Software-only (aes-generic):
#   Encrypt: ~200 MiB/s   Decrypt: ~210 MiB/s
```

### 5.4 Key Management with OP-TEE hwkey-agent

The critical question is: where does the LUKS passphrase/key come from? Storing
it in plaintext in the initrd defeats the purpose. The solution is to use the
OP-TEE hwkey-agent to derive a device-unique key:

```bash
#!/bin/bash
# /etc/initramfs-tools/scripts/local-top/unlock_rootfs.sh
# This script runs inside the initrd before rootfs is mounted

PREREQ=""
prereqs() { echo "$PREREQ"; }
case "$1" in
    prereqs) prereqs; exit 0 ;;
esac

# Start tee-supplicant in the background (needed for OP-TEE communication)
/usr/sbin/tee-supplicant &
sleep 1

# Use NVIDIA's ekb_unlock tool to derive the LUKS key from hardware
# The tool communicates with the hwkey-agent TA to derive a key using:
#   - Hardware Unique Key (HUK)
#   - EKB (Encrypted Key Blob) provisioned during manufacturing
/usr/bin/nvidia-luks-unlock \
    --ekb /etc/nvidia/ekb.dat \
    --label jetson-rootfs \
    --device /dev/mmcblk0p1

# nvidia-luks-unlock internally:
#   1. Opens a session with hwkey-agent TA
#   2. Sends the EKB blob to the TA
#   3. TA decrypts EKB using fuse-derived KEK2
#   4. TA returns the decrypted disk encryption key
#   5. Tool calls cryptsetup luksOpen with the key via --key-file

# Kill tee-supplicant -- systemd will restart it properly later
kill %1
```

### 5.5 Provisioning the Encrypted Key Blob (EKB)

The EKB is encrypted with a key derived from KEK2 fuse and contains the
actual LUKS disk encryption key:

```bash
# On the provisioning host

# 1. Generate a random 256-bit disk encryption key
openssl rand -hex 32 > disk_key.hex

# 2. Generate the EKB using NVIDIA's tool
python3 Linux_for_Tegra/tools/gen_ekb.py \
    --chip 0x23 \
    --key disk_key.hex \
    --fuse_key kek2.hex \
    --out ekb.dat

# The EKB structure:
# +------------------+
# | EKB Header       |  (version, magic, key count)
# +------------------+
# | Encrypted Key 1  |  (disk encryption key, encrypted with KEK2-derived key)
# +------------------+
# | Encrypted Key 2  |  (optional: additional keys)
# +------------------+
# | HMAC             |  (integrity check using KEK2-derived auth key)
# +------------------+

# 3. Flash the EKB to the EKB partition
sudo python3 bootloader/tegraflash.py \
    --chip 0x23 \
    --cmd "write EKB ekb.dat"
```

### 5.6 Encrypting Individual Partitions

Beyond the root filesystem, you may want to encrypt additional partitions:

```bash
# Encrypt a data partition for AI model storage
cryptsetup luksFormat \
    --type luks2 \
    --cipher aes-xts-plain64 \
    --key-size 512 \
    /dev/mmcblk0p12   # example data partition

# Add a key file (sealed by OP-TEE) as an additional LUKS key slot
# This allows automated unlocking via TA-derived key
cryptsetup luksAddKey /dev/mmcblk0p12 /tmp/sealed_key.bin

# fstab entry with automatic unlock via systemd-cryptsetup
# /etc/crypttab:
# ai-models  /dev/mmcblk0p12  /etc/keys/sealed_model_key.bin  luks,discard
```

### 5.7 Performance Impact and Tuning

```bash
# Measure encrypted I/O performance
# Sequential read
fio --name=seq_read --filename=/dev/mapper/jetson-rootfs \
    --rw=read --bs=1M --size=1G --numjobs=1 --direct=1

# Random 4K reads (typical for inference model loading)
fio --name=rand_read --filename=/dev/mapper/jetson-rootfs \
    --rw=randread --bs=4k --size=256M --numjobs=4 --direct=1

# Tuning: increase dm-crypt queue depth for better SE utilization
echo 64 > /sys/block/dm-0/queue/nr_requests

# Tuning: enable write-back caching if data loss risk is acceptable
cryptsetup --perf-no_read_workqueue --perf-no_write_workqueue \
    refresh jetson-rootfs
```

---

## 6. Fuse Programming

### 6.1 Fuse Overview

OTP fuses on the T234 are one-time-programmable bits. Once burned, they cannot
be reverted. Fuse programming is the foundational step that establishes the
hardware root of trust.

**WARNING:** Fuse burning is IRREVERSIBLE. A mistake in fuse programming can
permanently brick the device. Always validate on development units with
`--test` mode first.

### 6.2 Fuse Map for Security

```
T234 Security Fuse Map:
+-------------------------+--------+----------------------------------------+
| Fuse Name               | Bits   | Description                            |
+-------------------------+--------+----------------------------------------+
| public_key_hash         | 512    | SHA-512 of OEM PKC public key          |
| secure_boot_key (SBK)   | 128    | Symmetric encryption key for BL        |
| kek0                    | 128    | Key encryption key slot 0              |
| kek1                    | 128    | Key encryption key slot 1              |
| kek2                    | 256    | Key encryption key slot 2 (disk enc)   |
| security_mode           | 1      | Enable secure boot enforcement         |
| jtag_disable            | 1      | Disable JTAG debug port                |
| arm_jtag_disable        | 1      | Disable ARM CoreSight debug            |
| odm_lock                | 4      | Lock ODM fuse fields                   |
| odm_production_mode     | 1      | Enable production mode restrictions    |
| boot_security_info      | 6      | PKC key type and encryption config     |
| debug_authentication    | 5      | Authenticated debug configuration      |
+-------------------------+--------+----------------------------------------+
```

### 6.3 Reading Current Fuse State

```bash
# Method 1: Via sysfs (on a running Jetson)
cat /sys/devices/platform/efuse-burn/odm_production_mode
cat /sys/devices/platform/efuse-burn/security_mode
cat /sys/devices/platform/efuse-burn/jtag_disable

# Method 2: Via tegraflash on host (device in recovery mode)
cd Linux_for_Tegra
sudo python3 bootloader/tegraflash.py \
    --chip 0x23 \
    --applet mb1_t234_prod.bin \
    --cmd "readfuses fuses_readback.xml"

# Parse the readback
cat fuses_readback.xml
```

### 6.4 Generating Fuse Keys

```bash
# Generate a 128-bit SBK (Secure Boot Key)
openssl rand -hex 16 > sbk.key
# Example content: 0x12345678 0x9abcdef0 0x11223344 0x55667788

# Generate KEK0 (128-bit)
openssl rand -hex 16 > kek0.key

# Generate KEK1 (128-bit)
openssl rand -hex 16 > kek1.key

# Generate KEK2 (256-bit -- used for disk encryption key derivation)
openssl rand -hex 32 > kek2.key

# CRITICAL: Store these keys in a Hardware Security Module (HSM) or
# air-gapped vault. Loss of these keys means inability to update
# firmware on devices that have these fuses burned.
```

### 6.5 Fuse Configuration XML

```xml
<!-- fuse_config_t234.xml -->
<?xml version="1.0"?>
<genericfuse MagicId="0x45535546" version="1.0.0">
    <fuse name="PublicKeyHash" size="64" value="<512-bit hash from tegrasign>"/>
    <fuse name="SecureBootKey" size="16" value="0x12345678 0x9abcdef0 0x11223344 0x55667788"/>
    <fuse name="Kek0" size="16" value="<128-bit KEK0 value>"/>
    <fuse name="Kek1" size="16" value="<128-bit KEK1 value>"/>
    <fuse name="Kek256" size="32" value="<256-bit KEK2 value>"/>
    <fuse name="BootSecurityInfo" size="4" value="0x209"/>
        <!-- 0x209: RSA-3072 + AES-SBK encryption enabled -->
    <fuse name="SecurityMode" size="4" value="0x1"/>
    <fuse name="JtagDisable" size="4" value="0x1"/>
    <fuse name="ArmJtagDisable" size="4" value="0x1"/>
    <fuse name="OdmLock" size="4" value="0xF"/>
</genericfuse>
```

### 6.6 Fuse Burning Workflow

```bash
cd Linux_for_Tegra

# Step 1: DRY RUN -- Validate fuse config without actually burning
sudo python3 bootloader/tegraflash.py \
    --chip 0x23 \
    --applet mb1_t234_prod.bin \
    --cfg fuse_config_t234.xml \
    --cmd "blowfuses fuse_config_t234.xml --test"

# Step 2: BURN FUSES (IRREVERSIBLE)
# Ensure the device is in forced recovery mode (hold REC button during power-on)
sudo python3 bootloader/tegraflash.py \
    --chip 0x23 \
    --applet mb1_t234_prod.bin \
    --cfg fuse_config_t234.xml \
    --cmd "blowfuses fuse_config_t234.xml"

# Step 3: Verify the burned fuses
sudo python3 bootloader/tegraflash.py \
    --chip 0x23 \
    --applet mb1_t234_prod.bin \
    --cmd "readfuses fuses_verify.xml"

# Step 4: Compare expected vs actual
diff fuse_config_t234.xml fuses_verify.xml
```

### 6.7 Fuse Read-Back Protection

After fuses are programmed, you should enable read-back protection to prevent
software from reading the key fuse values:

```bash
# The OdmLock fuse bits control read protection:
#   Bit 0: Lock ODM_RESERVED fuses
#   Bit 1: Lock PKC/SBK read-back
#   Bit 2: Lock KEK0/KEK1 read-back
#   Bit 3: Lock KEK2 read-back
#
# Setting OdmLock to 0xF locks all key fuse read-back.
# The SE hardware key slots remain functional -- software just cannot
# read the raw fuse values.

# After burning OdmLock, verify that key readback returns zeros:
sudo python3 bootloader/tegraflash.py \
    --chip 0x23 \
    --applet mb1_t234_prod.bin \
    --cmd "readfuses fuses_locked.xml"
# SecureBootKey should read as 0x00000000 0x00000000 0x00000000 0x00000000
```

### 6.8 Production Fusing Workflow

```
Manufacturing Fuse Programming Sequence:

  +------------------+     +-------------------+     +------------------+
  | 1. Flash test FW |---->| 2. Board test     |---->| 3. Flash prod FW |
  | (unsigned)       |     | (ICT/FCT)         |     | (signed)         |
  +------------------+     +-------------------+     +--------+---------+
                                                              |
  +------------------+     +-------------------+     +--------v---------+
  | 6. Final verify  |<----| 5. Burn fuses     |<----| 4. Validate keys |
  | (boot signed FW) |     | (PKC+SBK+KEK+    |     | (test mode)      |
  +------------------+     |  security_mode)   |     +------------------+
                           +-------------------+

CRITICAL: Step 3 must happen BEFORE Step 5.
If you burn security_mode before flashing signed firmware, the device will
not boot and cannot be recovered via USB (bricked).
```

---

## 7. Secure Storage

### 7.1 Secure Storage Architecture

The T234 provides multiple layers of secure storage, each with different
security properties:

```
+-----------------------------------------------+
|           Secure Storage Layers                |
+-----------------------------------------------+
|                                                |
|  +------------------------------------------+ |
|  | Layer 3: OP-TEE Secure File Storage       | |
|  | - Files encrypted with TA-specific keys   | |
|  | - Stored in /data/tee/ on normal world FS | |
|  | - Integrity protected with HMAC           | |
|  +------------------------------------------+ |
|                                                |
|  +------------------------------------------+ |
|  | Layer 2: RPMB (Replay Protected Memory)   | |
|  | - eMMC hardware-enforced replay protect   | |
|  | - 4MB partition with auth read/write      | |
|  | - Used by OP-TEE for anti-rollback + keys | |
|  +------------------------------------------+ |
|                                                |
|  +------------------------------------------+ |
|  | Layer 1: OTP Fuses                        | |
|  | - Hardware Unique Key (HUK)               | |
|  | - Immutable after programming             | |
|  | - Root of all key derivation              | |
|  +------------------------------------------+ |
|                                                |
+-----------------------------------------------+
```

### 7.2 OP-TEE Secure Storage (REE-FS)

OP-TEE's default secure storage backend stores encrypted files on the normal
world filesystem (via `tee-supplicant`):

```c
/* Inside a Trusted Application -- secure file storage example */
#include <tee_internal_api.h>

TEE_Result store_secret(void *data, size_t data_len, const char *obj_id)
{
    TEE_Result res;
    TEE_ObjectHandle obj;
    uint32_t flags = TEE_DATA_FLAG_ACCESS_WRITE |
                     TEE_DATA_FLAG_ACCESS_READ |
                     TEE_DATA_FLAG_OVERWRITE;

    /* Create a persistent object in secure storage */
    res = TEE_CreatePersistentObject(
        TEE_STORAGE_PRIVATE,        /* Secure storage space */
        obj_id, strlen(obj_id),     /* Object identifier */
        flags,
        TEE_HANDLE_NULL,            /* No attributes */
        data, data_len,             /* Initial data */
        &obj);

    if (res == TEE_SUCCESS) {
        TEE_CloseObject(obj);
    }
    return res;
}

TEE_Result read_secret(const char *obj_id, void *buf, size_t *buf_len)
{
    TEE_Result res;
    TEE_ObjectHandle obj;
    uint32_t count;

    res = TEE_OpenPersistentObject(
        TEE_STORAGE_PRIVATE,
        obj_id, strlen(obj_id),
        TEE_DATA_FLAG_ACCESS_READ,
        &obj);
    if (res != TEE_SUCCESS) return res;

    res = TEE_ReadObjectData(obj, buf, *buf_len, &count);
    *buf_len = count;
    TEE_CloseObject(obj);
    return res;
}
```

The encrypted files are stored on the normal world filesystem:

```bash
# Location of OP-TEE secure storage files
ls -la /data/tee/

# Each file is named with a hash and contains:
#   - AES-GCM encrypted data
#   - IV (initialization vector)
#   - Tag (authentication tag)
#   - Metadata (encrypted with a different key)
#
# The encryption key is derived from:
#   - Hardware Unique Key (HUK)
#   - TA UUID (each TA gets its own key tree)
#   - File identifier
```

### 7.3 RPMB (Replay Protected Memory Block)

The eMMC on the Orin Nano includes an RPMB partition that provides
hardware-enforced authenticated writes with replay protection:

```bash
# Check RPMB partition availability
mmc rpmb read-counter /dev/mmcblk0rpmb

# RPMB properties on typical Orin Nano eMMC:
# - Size: 4 MB (configurable, max 16 MB)
# - Authentication: HMAC-SHA256 with a 256-bit key
# - Replay protection: Monotonic write counter (32-bit)
# - Access: Only via authenticated commands

# OP-TEE uses RPMB for:
#   1. Secure storage root encryption key tree
#   2. Anti-rollback counters for firmware
#   3. Critical security state flags
```

OP-TEE RPMB configuration in the device tree:

```bash
# Verify RPMB is configured for OP-TEE
dtc -I dtb -O dts /boot/dtb/kernel_tegra234-p3767-0003-p3768-0000-a0.dtb \
    2>/dev/null | grep -A10 "rpmb"
```

### 7.4 Hardware-Bound Key Derivation

```c
/* TA code: Derive a hardware-bound key using HUK */
TEE_Result derive_hw_bound_key(uint8_t *key_out, size_t key_len,
                                const char *context, size_t ctx_len)
{
    TEE_Result res;
    TEE_TASessionHandle hwkey_sess;
    TEE_UUID hwkey_uuid = { /* hwkey-agent UUID */ };
    TEE_Param params[4];
    uint32_t param_types;
    uint32_t err_origin;

    /* Alternative: Use TEE_GenerateKey with HKDF and HUK as input */
    /* This approach uses the OP-TEE internal derivation API */

    /* Method 1: Direct HUK derivation (OP-TEE extension) */
    uint8_t huk_derived[32];
    TEE_Result (*derive)(void *, size_t, const void *, size_t);

    /* The OP-TEE implementation provides tee_otp_get_hw_unique_key()
     * which derives a key from the HUK fuse using HMAC-SHA256 with
     * the provided context string as diversifier */

    param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_OUTPUT,  /* derived key */
        TEE_PARAM_TYPE_MEMREF_INPUT,   /* context/label */
        TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE
    );

    params[0].memref.buffer = key_out;
    params[0].memref.size = key_len;
    params[1].memref.buffer = (void *)context;
    params[1].memref.size = ctx_len;

    /* Open session with hwkey-agent */
    res = TEE_OpenTASession(&hwkey_uuid, 0, param_types,
                            params, &hwkey_sess, &err_origin);
    if (res != TEE_SUCCESS) return res;

    /* Invoke key derivation command */
    res = TEE_InvokeTACommand(hwkey_sess, 0,
                              HWKEY_CMD_DERIVE_KEY,
                              param_types, params,
                              &err_origin);

    TEE_CloseTASession(hwkey_sess);
    return res;
}
```

### 7.5 Secure Key Storage Best Practices

```
+----------------------------------------------------------------+
| Storage Type       | Use Case              | Security Level     |
+--------------------+-----------------------+--------------------+
| OTP Fuses          | Root keys (PKC, SBK,  | Highest -- HW      |
|                    | KEK), security config | immutable          |
+--------------------+-----------------------+--------------------+
| RPMB               | Anti-rollback ctr,    | High -- HW replay  |
|                    | root TA keys          | protected          |
+--------------------+-----------------------+--------------------+
| OP-TEE Secure FS   | API keys, tokens,     | Medium-High --     |
|                    | certificates, model   | encrypted + HMAC   |
|                    | decryption keys       | but on REE storage |
+--------------------+-----------------------+--------------------+
| LUKS-encrypted FS  | Bulk data, models,    | Medium -- depends  |
|                    | logs                  | on key management  |
+--------------------+-----------------------+--------------------+
| Plaintext FS       | Public configs,       | None               |
|                    | non-sensitive data    |                    |
+--------------------+-----------------------+--------------------+
```

---

## 8. Network Security

### 8.1 TLS Configuration

Edge devices communicating with cloud services or fleet management backends
must enforce strong TLS:

```bash
# /etc/ssl/openssl_jetson.cnf -- Hardened OpenSSL configuration
[system_default_sect]
MinProtocol = TLSv1.2
CipherString = ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256
Ciphersuites = TLS_AES_256_GCM_SHA384:TLS_AES_128_GCM_SHA256
Options = ServerPreference,PrioritizeChaCha,NoSSLv3,NoTLSv1,NoTLSv1.1
```

```bash
# Apply the hardened TLS config system-wide
export OPENSSL_CONF=/etc/ssl/openssl_jetson.cnf

# Test TLS connection to management server
openssl s_client -connect fleet-mgmt.example.com:443 \
    -tls1_3 \
    -CAfile /etc/ssl/certs/fleet-ca.pem \
    -cert /etc/ssl/certs/device.pem \
    -key /etc/ssl/private/device.key
```

### 8.2 Mutual TLS (mTLS) for Device Authentication

Each Jetson device should have a unique client certificate provisioned during
manufacturing:

```bash
# During manufacturing: Generate per-device key pair and CSR
openssl ecparam -genkey -name prime256v1 -out /etc/ssl/private/device.key
chmod 600 /etc/ssl/private/device.key

# Generate CSR with device-specific identity
openssl req -new -key /etc/ssl/private/device.key \
    -out /tmp/device.csr \
    -subj "/CN=jetson-${SERIAL_NUMBER}/O=OEM Corp/OU=Edge Devices"

# Sign with the OEM CA (done on the provisioning server)
openssl x509 -req -in /tmp/device.csr \
    -CA oem_ca.crt -CAkey oem_ca.key \
    -CAcreateserial \
    -out /etc/ssl/certs/device.pem \
    -days 3650 \
    -sha256

# Store the private key in OP-TEE secure storage for production
# (instead of filesystem -- see Section 7)
```

### 8.3 Certificate Management and Rotation

```bash
#!/bin/bash
# /usr/local/bin/cert_rotation.sh
# Automated certificate rotation script

DEVICE_KEY="/etc/ssl/private/device.key"
DEVICE_CERT="/etc/ssl/certs/device.pem"
CA_CERT="/etc/ssl/certs/fleet-ca.pem"
RENEWAL_SERVER="https://pki.example.com/renew"

# Check certificate expiry (renew if less than 30 days remaining)
EXPIRY=$(openssl x509 -in "$DEVICE_CERT" -enddate -noout | cut -d= -f2)
EXPIRY_EPOCH=$(date -d "$EXPIRY" +%s)
NOW_EPOCH=$(date +%s)
DAYS_LEFT=$(( (EXPIRY_EPOCH - NOW_EPOCH) / 86400 ))

if [ "$DAYS_LEFT" -lt 30 ]; then
    echo "Certificate expires in $DAYS_LEFT days, renewing..."

    # Generate new CSR
    openssl req -new -key "$DEVICE_KEY" -out /tmp/renew.csr \
        -subj "$(openssl x509 -in "$DEVICE_CERT" -subject -noout | sed 's/subject=//')"

    # Submit CSR to PKI server using current (still valid) mTLS cert
    curl -s --cert "$DEVICE_CERT" --key "$DEVICE_KEY" --cacert "$CA_CERT" \
        -X POST -d @/tmp/renew.csr \
        "$RENEWAL_SERVER" -o /tmp/new_cert.pem

    # Validate new certificate before replacing
    if openssl verify -CAfile "$CA_CERT" /tmp/new_cert.pem; then
        cp /tmp/new_cert.pem "$DEVICE_CERT"
        systemctl restart fleet-agent
        echo "Certificate renewed successfully"
    else
        echo "ERROR: New certificate failed validation"
    fi
    rm -f /tmp/renew.csr /tmp/new_cert.pem
fi
```

### 8.4 Firewall Configuration

```bash
# /etc/nftables.conf -- Production firewall rules for Jetson Orin Nano
#!/usr/sbin/nft -f

flush ruleset

table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        # Allow loopback
        iif lo accept

        # Allow established/related connections
        ct state established,related accept

        # Rate-limited ICMP (for health checks)
        ip protocol icmp limit rate 10/second accept
        ip6 nexthdr icmpv6 limit rate 10/second accept

        # SSH -- restrict to management VLAN only
        tcp dport 22 ip saddr 10.0.100.0/24 accept

        # Fleet management agent (outbound-initiated, but allow return)
        # No inbound ports needed if agent initiates connection

        # MQTT for telemetry (if using local broker)
        # tcp dport 8883 ip saddr 10.0.100.0/24 accept

        # Drop everything else with logging
        log prefix "[nft-drop] " limit rate 5/minute
        drop
    }

    chain forward {
        type filter hook forward priority 0; policy drop;
        # No forwarding on edge device
    }

    chain output {
        type filter hook output priority 0; policy accept;

        # Restrict outbound to known services
        # DNS
        udp dport 53 accept
        tcp dport 53 accept

        # NTP
        udp dport 123 accept

        # HTTPS (fleet management, OTA updates, telemetry)
        tcp dport 443 accept

        # MQTT over TLS (telemetry)
        tcp dport 8883 accept

        # Block all other outbound
        # (Uncomment for strict production lockdown)
        # drop
    }
}

# Alternative: iptables equivalent for systems without nftables
# iptables -P INPUT DROP
# iptables -P FORWARD DROP
# iptables -A INPUT -i lo -j ACCEPT
# iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
# iptables -A INPUT -p tcp --dport 22 -s 10.0.100.0/24 -j ACCEPT
# iptables -A INPUT -j LOG --log-prefix "[iptables-drop] "
# iptables -A INPUT -j DROP
```

```bash
# Enable and start nftables
sudo systemctl enable nftables
sudo systemctl start nftables

# Verify rules are loaded
sudo nft list ruleset
```

### 8.5 VPN for Fleet Management

```bash
# WireGuard VPN configuration for fleet management
# Preferred over OpenVPN for performance on resource-constrained devices

# Install WireGuard
sudo apt-get install wireguard

# Generate device keypair
wg genkey | tee /etc/wireguard/device_private.key | wg pubkey > /etc/wireguard/device_public.key
chmod 600 /etc/wireguard/device_private.key

# /etc/wireguard/wg-fleet.conf
cat > /etc/wireguard/wg-fleet.conf << 'WGEOF'
[Interface]
Address = 10.200.0.100/32
PrivateKey = <device_private_key>
DNS = 10.200.0.1

# Keep-alive to maintain NAT mappings in field deployments
[Peer]
PublicKey = <fleet_server_public_key>
Endpoint = vpn.example.com:51820
AllowedIPs = 10.200.0.0/24
PersistentKeepalive = 25
WGEOF

chmod 600 /etc/wireguard/wg-fleet.conf

# Enable WireGuard on boot
sudo systemctl enable wg-quick@wg-fleet
sudo systemctl start wg-quick@wg-fleet

# Verify tunnel
wg show wg-fleet
```

### 8.6 Network Hardening Checklist

```bash
# Disable unused network services
sudo systemctl disable avahi-daemon
sudo systemctl disable cups-browsed
sudo systemctl stop avahi-daemon

# Disable IPv6 if not needed (reduces attack surface)
echo "net.ipv6.conf.all.disable_ipv6 = 1" >> /etc/sysctl.d/99-security.conf
echo "net.ipv6.conf.default.disable_ipv6 = 1" >> /etc/sysctl.d/99-security.conf

# Harden TCP/IP stack
cat >> /etc/sysctl.d/99-security.conf << 'EOF'
# Prevent SYN flood
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048

# Disable source routing
net.ipv4.conf.all.accept_source_route = 0

# Disable ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0

# Enable reverse path filtering (anti-spoofing)
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Log martian packets
net.ipv4.conf.all.log_martians = 1

# Disable IP forwarding
net.ipv4.ip_forward = 0
EOF

sudo sysctl --system
```

---

## 9. Firmware Update Security

### 9.1 OTA Update Threat Model

Over-the-air firmware updates are the primary post-deployment attack vector.
Without proper signing and verification, an attacker who compromises the update
server or performs a MitM attack can push malicious firmware to the entire fleet.

```
Threats to OTA Updates:
  1. Unsigned payload    --> Attacker injects malicious firmware
  2. Replay attack       --> Attacker pushes old vulnerable firmware
  3. Partial update      --> Power loss mid-update bricks device
  4. Server compromise   --> All fleet devices receive malicious update
  5. Downgrade attack    --> Attacker forces rollback to bypass patches
```

### 9.2 Signed OTA Update Architecture

```
+------------------+         +-------------------+        +----------------+
| Build Server     |         | OTA Distribution  |        | Jetson Device  |
|                  |         | Server            |        |                |
| 1. Build FW      |         |                   |        |                |
| 2. Sign with     |-------->| 3. Host signed    |------->| 4. Download    |
|    OEM key       |         |    payload +      |        | 5. Verify sig  |
| 2a. Generate     |         |    manifest       |        | 6. Check ver   |
|     manifest     |         |                   |        | 7. Apply to B  |
|                  |         |                   |        | 8. Mark B good |
+------------------+         +-------------------+        +----------------+
```

### 9.3 Update Manifest Format

```json
{
    "version": "36.4.1-20260301",
    "build_id": "a1b2c3d4e5f6",
    "timestamp": "2026-03-01T12:00:00Z",
    "min_version": "36.3.0",
    "target_board": "jetson-orin-nano-devkit",
    "target_sku": "p3767-0003",
    "components": [
        {
            "name": "mb1_t234_prod.bin",
            "partition": "mb1",
            "sha256": "a1b2c3...hex_hash",
            "size": 262144
        },
        {
            "name": "uefi_jetson_with_dtb.bin",
            "partition": "esp",
            "sha256": "d4e5f6...hex_hash",
            "size": 4194304
        },
        {
            "name": "boot.img",
            "partition": "kernel",
            "sha256": "789abc...hex_hash",
            "size": 67108864
        }
    ],
    "signature": "<RSA-PSS signature of the above fields with OEM key>"
}
```

### 9.4 Update Verification Script

```bash
#!/bin/bash
# /usr/local/bin/ota_verify.sh -- Verify OTA update before applying
set -euo pipefail

MANIFEST="$1"
PAYLOAD_DIR="$2"
OEM_PUBKEY="/etc/nvidia/ota_verify.pub"
CURRENT_VERSION=$(cat /etc/nvidia/fw_version)

echo "=== OTA Update Verification ==="

# Step 1: Verify manifest signature
echo "[1/5] Verifying manifest signature..."
MANIFEST_DATA=$(jq -r 'del(.signature)' "$MANIFEST")
MANIFEST_SIG=$(jq -r '.signature' "$MANIFEST")
echo "$MANIFEST_SIG" | base64 -d > /tmp/manifest.sig
echo "$MANIFEST_DATA" | openssl dgst -sha256 -verify "$OEM_PUBKEY" \
    -signature /tmp/manifest.sig
if [ $? -ne 0 ]; then
    echo "FATAL: Manifest signature verification FAILED"
    exit 1
fi
echo "    Manifest signature: VALID"

# Step 2: Check version (anti-rollback)
echo "[2/5] Checking version..."
NEW_VERSION=$(jq -r '.version' "$MANIFEST")
MIN_VERSION=$(jq -r '.min_version' "$MANIFEST")
if dpkg --compare-versions "$CURRENT_VERSION" lt "$MIN_VERSION"; then
    echo "FATAL: Current version $CURRENT_VERSION is below minimum $MIN_VERSION"
    exit 1
fi
if dpkg --compare-versions "$NEW_VERSION" le "$CURRENT_VERSION"; then
    echo "FATAL: Downgrade attempt blocked ($NEW_VERSION <= $CURRENT_VERSION)"
    exit 1
fi
echo "    Version check: $CURRENT_VERSION -> $NEW_VERSION (OK)"

# Step 3: Verify target board
echo "[3/5] Checking target compatibility..."
TARGET_SKU=$(jq -r '.target_sku' "$MANIFEST")
DEVICE_SKU=$(cat /proc/device-tree/nvidia,dtsfilename 2>/dev/null | \
    sed 's/.*\(p3767-[0-9]*\).*/\1/')
if [ "$TARGET_SKU" != "$DEVICE_SKU" ]; then
    echo "FATAL: SKU mismatch (target: $TARGET_SKU, device: $DEVICE_SKU)"
    exit 1
fi
echo "    SKU match: $TARGET_SKU (OK)"

# Step 4: Verify component hashes
echo "[4/5] Verifying component checksums..."
COMPONENTS=$(jq -r '.components[] | "\(.name) \(.sha256)"' "$MANIFEST")
while IFS=' ' read -r name expected_hash; do
    actual_hash=$(sha256sum "$PAYLOAD_DIR/$name" | cut -d' ' -f1)
    if [ "$actual_hash" != "$expected_hash" ]; then
        echo "FATAL: Hash mismatch for $name"
        echo "  Expected: $expected_hash"
        echo "  Actual:   $actual_hash"
        exit 1
    fi
    echo "    $name: VERIFIED"
done <<< "$COMPONENTS"

# Step 5: Check available space
echo "[5/5] Checking storage space..."
TOTAL_SIZE=$(jq '[.components[].size] | add' "$MANIFEST")
AVAIL=$(df -B1 /tmp | tail -1 | awk '{print $4}')
if [ "$AVAIL" -lt "$TOTAL_SIZE" ]; then
    echo "FATAL: Insufficient space (need $TOTAL_SIZE, have $AVAIL)"
    exit 1
fi
echo "    Storage: OK"

echo "=== All checks passed. Update is safe to apply. ==="
```

### 9.5 Anti-Rollback Protection

The T234 supports hardware anti-rollback via RPMB-stored rollback counters:

```bash
# Check current rollback counter
# The rollback counter is stored in RPMB and managed by MB2/OP-TEE

# Anti-rollback is enforced at multiple levels:
#
# Level 1: Boot component rollback index (burned into BCT)
#   - Each signed BCT contains a rollback index
#   - MB1 compares BCT index against RPMB-stored minimum
#   - If BCT index < stored minimum, boot is rejected
#
# Level 2: Kernel/rootfs rollback
#   - Managed by the OTA update agent
#   - Software-enforced version comparison
#   - Backed by RPMB counter increment after successful boot

# Increment rollback counter after verified boot (in update agent)
# This is typically done by OP-TEE TA:
# 1. Update agent verifies new FW boots successfully
# 2. Sends "commit" command to anti-rollback TA
# 3. TA increments RPMB counter
# 4. Old firmware can no longer be booted

# NVIDIA's nv_update_engine handles this automatically:
sudo nv_update_engine --verify-and-commit
```

### 9.6 A/B Partition Update Integrity

The Orin Nano supports A/B redundancy for safe updates:

```bash
# Check current boot slot
sudo nvbootctrl get-current-slot
# Output: Current boot slot: A (or B)

# View partition layout for A/B slots
sudo nvbootctrl dump-slots-info
# Output:
# Slot A:
#   Priority: 15
#   Status: normal
#   Retry count: 7
# Slot B:
#   Priority: 14
#   Status: unbootable
#   Retry count: 0

# OTA update flow with A/B:
# 1. Identify inactive slot
INACTIVE=$(sudo nvbootctrl get-current-slot | grep -q "A" && echo "B" || echo "A")
echo "Updating slot: $INACTIVE"

# 2. Write verified update to inactive slot partitions
# (using NVIDIA's nv_update_engine or custom updater)
sudo nv_update_engine \
    --slot "$INACTIVE" \
    --payload /tmp/ota_payload/ \
    --manifest /tmp/ota_manifest.json

# 3. Mark inactive slot as bootable and set higher priority
sudo nvbootctrl set-slot-as-unbootable "$INACTIVE"  # reset state
sudo nvbootctrl mark-boot-successful "$INACTIVE"     # mark good

# 4. Set next boot to new slot
sudo nvbootctrl set-active-boot-slot "$INACTIVE"

# 5. Reboot into new slot
sudo reboot

# 6. After successful boot, the watchdog / health check confirms,
#    and the rollback counter is incremented
```

---

## 10. Runtime Security

### 10.1 seccomp Profiles

seccomp (Secure Computing Mode) restricts the system calls available to a
process. This is critical for inference workloads:

```json
{
    "defaultAction": "SCMP_ACT_ERRNO",
    "defaultErrnoRet": 1,
    "archMap": [
        {
            "architecture": "SCMP_ARCH_AARCH64",
            "subArchitectures": []
        }
    ],
    "syscalls": [
        {
            "names": [
                "read", "write", "close", "fstat", "lseek",
                "mmap", "mprotect", "munmap", "brk",
                "ioctl", "openat", "newfstatat",
                "clone", "execve", "exit_group",
                "futex", "nanosleep", "clock_gettime",
                "getpid", "gettid", "tgkill",
                "rt_sigaction", "rt_sigprocmask",
                "sched_yield", "sched_getaffinity",
                "getrandom"
            ],
            "action": "SCMP_ACT_ALLOW"
        },
        {
            "comment": "Allow GPU ioctls for inference",
            "names": ["ioctl"],
            "action": "SCMP_ACT_ALLOW",
            "args": [
                {
                    "index": 1,
                    "value": 1074029824,
                    "op": "SCMP_CMP_MASKED_EQ",
                    "valueTwo": 4278190080
                }
            ]
        }
    ]
}
```

```bash
# Apply seccomp profile to an inference container
docker run --security-opt seccomp=/etc/seccomp/inference_profile.json \
    --runtime nvidia \
    nvcr.io/nvidia/l4t-tensorrt:r8.6

# Or apply to a systemd service
# /etc/systemd/system/inference.service
# [Service]
# SystemCallFilter=@system-service
# SystemCallFilter=~@mount @reboot @swap @raw-io
# SystemCallErrorNumber=EPERM
```

### 10.2 AppArmor on Jetson

L4T ships with AppArmor enabled by default. Custom profiles should be created
for inference workloads:

```bash
# Check AppArmor status
sudo aa-status

# Example AppArmor profile for an inference application
# /etc/apparmor.d/usr.local.bin.inference_engine
cat > /etc/apparmor.d/usr.local.bin.inference_engine << 'APPARMOR_EOF'
#include <tunables/global>

/usr/local/bin/inference_engine {
    #include <abstractions/base>
    #include <abstractions/nameservice>

    # Allow reading model files (encrypted partition)
    /opt/models/** r,

    # Allow GPU device access
    /dev/nvhost-* rw,
    /dev/nvmap rw,
    /dev/nvgpu/* rw,
    /dev/dri/* rw,

    # Allow DLA device access
    /dev/nvdla* rw,

    # Allow camera access (if needed for inference pipeline)
    /dev/video* rw,

    # TensorRT and CUDA libraries
    /usr/lib/aarch64-linux-gnu/tegra/** mr,
    /usr/local/cuda/lib64/** mr,
    /usr/lib/aarch64-linux-gnu/libnv*.so* mr,

    # Deny network access (inference-only process)
    deny network inet,
    deny network inet6,

    # Deny ptrace (anti-debugging)
    deny ptrace,

    # Allow writing inference results to specific directory
    /var/run/inference/** rw,

    # Deny everything else by default (implicit in AppArmor)
}
APPARMOR_EOF

# Load the profile
sudo apparmor_parser -r /etc/apparmor.d/usr.local.bin.inference_engine

# Enforce the profile
sudo aa-enforce /etc/apparmor.d/usr.local.bin.inference_engine
```

### 10.3 Namespace and cgroup Isolation

```bash
# Run inference workload in an isolated namespace
# Using systemd for production service management

# /etc/systemd/system/inference-engine.service
cat > /etc/systemd/system/inference-engine.service << 'SVCEOF'
[Unit]
Description=AI Inference Engine
After=network.target tee-supplicant.service

[Service]
Type=simple
ExecStart=/usr/local/bin/inference_engine --config /etc/inference/config.yaml

# Namespace isolation
PrivateNetwork=no
PrivateTmp=yes
PrivateDevices=no
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/run/inference /opt/models
ReadOnlyPaths=/usr/local/cuda /usr/lib/aarch64-linux-gnu/tegra

# Device access whitelist
DeviceAllow=/dev/nvhost-ctrl rw
DeviceAllow=/dev/nvhost-ctrl-gpu rw
DeviceAllow=/dev/nvhost-gpu rw
DeviceAllow=/dev/nvhost-as-gpu rw
DeviceAllow=/dev/nvmap rw
DeviceAllow=/dev/nvdla0 rw
DeviceAllow=/dev/nvdla1 rw

# Resource limits
MemoryMax=4G
CPUQuota=400%
TasksMax=128

# Capability dropping
CapabilityBoundingSet=
AmbientCapabilities=
NoNewPrivileges=yes

# seccomp
SystemCallFilter=@system-service
SystemCallFilter=~@mount @reboot @swap @raw-io @clock @module @debug
SystemCallErrorNumber=EPERM

# Security
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictNamespaces=yes
RestrictRealtime=yes
LockPersonality=yes
MemoryDenyWriteExecute=no   # Required for JIT in CUDA runtime

[Install]
WantedBy=multi-user.target
SVCEOF

sudo systemctl daemon-reload
sudo systemctl enable inference-engine
```

### 10.4 Capability Dropping

```bash
# List current capabilities of a running process
cat /proc/$(pgrep inference_engine)/status | grep Cap

# Decode capabilities
capsh --decode=00000000a80425fb

# For inference workloads, drop ALL capabilities:
# In the application itself (C code):
```

```c
/* capability_drop.c -- Drop all capabilities at startup */
#include <sys/capability.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <stdio.h>

int drop_all_capabilities(void)
{
    cap_t caps;

    /* Clear all capabilities */
    caps = cap_init();
    if (caps == NULL) {
        perror("cap_init");
        return -1;
    }

    if (cap_set_proc(caps) != 0) {
        perror("cap_set_proc");
        cap_free(caps);
        return -1;
    }
    cap_free(caps);

    /* Prevent regaining capabilities */
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        perror("prctl NO_NEW_PRIVS");
        return -1;
    }

    /* Lock securebits to prevent capability acquisition */
    if (prctl(PR_SET_SECUREBITS,
              SECBIT_KEEP_CAPS_LOCKED |
              SECBIT_NO_SETUID_FIXUP |
              SECBIT_NO_SETUID_FIXUP_LOCKED |
              SECBIT_NOROOT |
              SECBIT_NOROOT_LOCKED, 0, 0, 0) != 0) {
        perror("prctl SECUREBITS");
        return -1;
    }

    printf("All capabilities dropped successfully\n");
    return 0;
}

int main(int argc, char *argv[])
{
    /* Initialize GPU / TensorRT before dropping capabilities */
    /* init_tensorrt(); */

    /* Drop capabilities after initialization */
    if (drop_all_capabilities() != 0) {
        fprintf(stderr, "Failed to drop capabilities, aborting\n");
        return 1;
    }

    /* Run inference loop with minimal privileges */
    /* inference_loop(); */

    return 0;
}
```

### 10.5 Container Security for Jetson

```bash
# Secure Docker daemon configuration for Jetson
# /etc/docker/daemon.json
cat > /etc/docker/daemon.json << 'DOCKEREOF'
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia",
    "storage-driver": "overlay2",
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "live-restore": true,
    "userns-remap": "default",
    "no-new-privileges": true,
    "icc": false,
    "userns-remap": "dockremap",
    "seccomp-profile": "/etc/docker/seccomp-default.json"
}
DOCKEREOF

# Run inference container with security restrictions
docker run -d \
    --name inference \
    --runtime nvidia \
    --gpus all \
    --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,size=256m \
    --security-opt no-new-privileges:true \
    --security-opt apparmor=docker-inference \
    --cap-drop ALL \
    --cap-add SYS_NICE \
    --pids-limit 100 \
    --memory 4g \
    --memory-swap 4g \
    --cpus 4 \
    -v /opt/models:/models:ro \
    -v /var/run/inference:/output:rw \
    inference-image:latest
```

---

## 11. Model Protection

### 11.1 AI Model Threat Landscape

Proprietary AI models represent significant IP investment. On an edge device,
models face:

| Threat                 | Attack Vector                              |
|------------------------|--------------------------------------------|
| Model extraction       | Read model file from disk/eMMC             |
| Weight theft           | Dump GPU memory during inference           |
| Architecture reversal  | Analyze model structure from ONNX/TRT file |
| Training data leakage  | Membership inference on deployed model     |
| Model substitution     | Replace model with adversarial version     |

### 11.2 Encrypted Model Storage

```bash
# Encrypt model files before deployment using a device-class key
# The actual decryption key is sealed per-device via OP-TEE

# On the build server: encrypt model
openssl enc -aes-256-cbc -salt -pbkdf2 \
    -in resnet50_jetson.engine \
    -out resnet50_jetson.engine.enc \
    -pass file:model_encryption_key.bin

# The model_encryption_key.bin is then encrypted per-device
# using the device's public key or via EKB mechanism (see Section 5.5)
```

### 11.3 Model Decryption via OP-TEE TA

```c
/* model_decrypt_ta.c -- Trusted Application for model decryption */
#include <tee_internal_api.h>

#define TA_MODEL_DECRYPT_UUID \
    { 0xdeadbeef, 0x1234, 0x5678, \
      { 0x9a, 0xbc, 0xde, 0xf0, 0x12, 0x34, 0x56, 0x78 } }

#define CMD_DECRYPT_MODEL_KEY  0
#define CMD_DECRYPT_CHUNK      1

/* Session state holds the decrypted model key */
struct session_ctx {
    TEE_OperationHandle aes_op;
    uint8_t model_key[32];
    bool key_loaded;
};

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                     TEE_Param params[4],
                                     void **session_ctx)
{
    struct session_ctx *ctx = TEE_Malloc(sizeof(*ctx), TEE_MALLOC_FILL_ZERO);
    if (!ctx) return TEE_ERROR_OUT_OF_MEMORY;
    ctx->key_loaded = false;
    *session_ctx = ctx;
    return TEE_SUCCESS;
}

static TEE_Result decrypt_model_key(struct session_ctx *ctx,
                                     uint32_t param_types,
                                     TEE_Param params[4])
{
    TEE_Result res;
    uint8_t hw_key[32];
    size_t hw_key_len = sizeof(hw_key);

    /* Derive hardware-bound key from HUK for model decryption */
    /* Uses hwkey-agent TA (see Section 7.4) */
    const char label[] = "model-decryption-key-v1";

    /* ... derive hw_key using hwkey-agent ... */

    /* Decrypt the encrypted model key using the HW-derived key */
    void *encrypted_mkey = params[0].memref.buffer;
    size_t emkey_len = params[0].memref.size;

    TEE_OperationHandle dec_op;
    res = TEE_AllocateOperation(&dec_op, TEE_ALG_AES_GCM,
                                 TEE_MODE_DECRYPT, 256);
    if (res != TEE_SUCCESS) return res;

    /* Set up key and decrypt */
    TEE_ObjectHandle key_obj;
    TEE_Attribute attr;
    TEE_InitRefAttribute(&attr, TEE_ATTR_SECRET_VALUE, hw_key, 32);
    res = TEE_AllocateTransientObject(TEE_TYPE_AES, 256, &key_obj);
    if (res != TEE_SUCCESS) goto out;

    res = TEE_PopulateTransientObject(key_obj, &attr, 1);
    if (res != TEE_SUCCESS) goto out;

    res = TEE_SetOperationKey(dec_op, key_obj);
    if (res != TEE_SUCCESS) goto out;

    /* Extract IV from encrypted blob header */
    uint8_t *iv = encrypted_mkey;
    TEE_AEInit(dec_op, iv, 12, 128, 0);

    size_t mkey_len = sizeof(ctx->model_key);
    res = TEE_AEDecryptFinal(dec_op,
                              (uint8_t *)encrypted_mkey + 12,
                              emkey_len - 12 - 16,
                              ctx->model_key, &mkey_len,
                              (uint8_t *)encrypted_mkey + emkey_len - 16,
                              16);
    if (res == TEE_SUCCESS) {
        ctx->key_loaded = true;
    }

out:
    TEE_FreeOperation(dec_op);
    TEE_FreeTransientObject(key_obj);
    TEE_MemFill(hw_key, 0, sizeof(hw_key));
    return res;
}

TEE_Result TA_InvokeCommandEntryPoint(void *session,
                                       uint32_t cmd,
                                       uint32_t param_types,
                                       TEE_Param params[4])
{
    struct session_ctx *ctx = session;
    switch (cmd) {
    case CMD_DECRYPT_MODEL_KEY:
        return decrypt_model_key(ctx, param_types, params);
    case CMD_DECRYPT_CHUNK:
        if (!ctx->key_loaded)
            return TEE_ERROR_BAD_STATE;
        /* Decrypt a chunk of model data using the loaded key */
        /* Stream decryption for large models that do not fit in TA memory */
        return TEE_ERROR_NOT_IMPLEMENTED; /* implement as needed */
    default:
        return TEE_ERROR_BAD_PARAMETERS;
    }
}
```

### 11.4 Runtime Model Integrity Verification

```bash
#!/bin/bash
# /usr/local/bin/verify_model.sh
# Verify model integrity before loading

MODEL_PATH="$1"
MODEL_MANIFEST="/opt/models/manifest.json"
OEM_PUBKEY="/etc/nvidia/model_signing.pub"

# Read expected hash from signed manifest
EXPECTED_HASH=$(jq -r --arg m "$(basename "$MODEL_PATH")" \
    '.models[] | select(.name == $m) | .sha256' "$MODEL_MANIFEST")

# Compute actual hash
ACTUAL_HASH=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)

if [ "$EXPECTED_HASH" != "$ACTUAL_HASH" ]; then
    echo "ERROR: Model integrity check failed for $MODEL_PATH"
    echo "  Expected: $EXPECTED_HASH"
    echo "  Actual:   $ACTUAL_HASH"
    logger -p auth.alert "Model integrity failure: $MODEL_PATH"
    exit 1
fi

echo "Model integrity verified: $MODEL_PATH"
```

### 11.5 Model Loading Architecture

```
Secure Model Loading Flow:

  +-------------------+
  | Encrypted model   |   (on LUKS-encrypted /opt/models/)
  | file (.engine.enc)|
  +--------+----------+
           |
  +--------v----------+
  | CA: Request model  |   (Linux user-space inference app)
  | key from OP-TEE   |
  +--------+----------+
           |  TEE Client API (TEEC_InvokeCommand)
  +--------v----------+
  | TA: Decrypt model  |   (Secure world TA)
  | key using HUK-     |
  | derived key        |
  +--------+----------+
           |  Returns decrypted model key to shared memory
  +--------v----------+
  | CA: Decrypt model  |   (Uses SE hardware AES via /dev/tegra-se)
  | in chunks, stream  |
  | to GPU memory      |
  +--------+----------+
           |
  +--------v----------+
  | GPU: Model loaded  |   (TensorRT engine in GPU DRAM)
  | in plaintext only  |
  | in volatile memory |
  +-------------------+

  On power loss, the plaintext model in GPU memory is lost.
  The encrypted copy on disk remains protected.
```

---

## 12. Debug Security

### 12.1 JTAG/SWD Lock-Down

JTAG provides full hardware debug access including memory reads, register
dumps, and code injection. It must be disabled in production.

```bash
# Check JTAG disable status
cat /sys/devices/platform/efuse-burn/jtag_disable
# 0 = JTAG enabled (development)
# 1 = JTAG disabled (production)

# The jtag_disable fuse is burned as part of the production fusing
# workflow (see Section 6.6).
#
# IMPORTANT: Disable JTAG as the LAST fusing step, after verifying
# that signed firmware boots correctly. If JTAG is disabled on a
# device that cannot boot, recovery is impossible.

# The T234 supports three JTAG security levels:
#
# Level 0: Full JTAG access (development)
# Level 1: Authenticated JTAG (requires challenge-response)
# Level 2: JTAG completely disabled (production)
#
# Level 1 is configured via the debug_authentication fuse:
# This allows authorized engineers to debug production units
# using a challenge-response protocol with a private key.
```

### 12.2 Authenticated Debug (Level 1)

```bash
# Configure authenticated debug (alternative to full JTAG disable)
# This is useful for field debugging without full JTAG exposure

# 1. Generate debug authentication key pair
openssl ecparam -genkey -name prime256v1 -out debug_auth.key
openssl ec -in debug_auth.key -pubout -out debug_auth.pub

# 2. Hash the debug public key for fuse programming
python3 bootloader/tegrasign_v3.py \
    --pubkeyhash debug_auth_hash.bin \
    --key debug_auth.key

# 3. Program the debug authentication fuse
# In fuse_config_t234.xml:
# <fuse name="DebugAuthentication" size="4" value="0x01"/>
# <fuse name="DebugAuthKeyHash" size="32" value="<hash>"/>

# 4. At debug time: challenge-response via NVIDIA debug tool
# The debug tool connects via JTAG/SWD, receives a nonce from
# the T234, signs it with the debug private key, and sends it back.
# Only if the signature verifies against the fused hash does
# the debug port become active.
```

### 12.3 UART Console Security

```bash
# Disable UART console output in production
# Method 1: Kernel command line
# In extlinux.conf or UEFI boot config:
#   APPEND ... console= loglevel=0 quiet

# Method 2: Disable serial getty
sudo systemctl disable serial-getty@ttyTCU0.service
sudo systemctl stop serial-getty@ttyTCU0.service
sudo systemctl mask serial-getty@ttyTCU0.service

# Method 3: Remove console from device tree
# In the device tree overlay:
# /delete-node/ serial@3100000;  /* UART0 (debug console) */

# Method 4: Password-protect UEFI console (if console must remain)
# In UEFI configuration, set a console password:
# This prevents unauthorized access to UEFI shell
```

### 12.4 Disabling Debug Interfaces Comprehensively

```bash
# Production debug lockdown script
#!/bin/bash
# /usr/local/bin/production_debug_lockdown.sh

echo "=== Production Debug Interface Lockdown ==="

# 1. Disable kernel debug filesystem
echo "[1] Disabling debugfs..."
if mountpoint -q /sys/kernel/debug; then
    umount /sys/kernel/debug
fi
# Prevent future mounts:
echo "debugfs /sys/kernel/debug debugfs noauto 0 0" >> /etc/fstab

# 2. Disable tracefs
echo "[2] Disabling tracefs..."
if mountpoint -q /sys/kernel/tracing; then
    umount /sys/kernel/tracing
fi

# 3. Disable kprobes and ftrace
echo "[3] Disabling kernel tracing..."
echo 0 > /proc/sys/kernel/kptr_restrict || true
echo 2 > /proc/sys/kernel/kptr_restrict  # Hide kernel pointers
echo 1 > /proc/sys/kernel/dmesg_restrict  # Restrict dmesg to root
echo 0 > /proc/sys/kernel/perf_event_paranoid || true
echo 3 > /proc/sys/kernel/perf_event_paranoid  # Disable perf events

# 4. Disable core dumps
echo "[4] Disabling core dumps..."
echo "kernel.core_pattern=|/bin/false" >> /etc/sysctl.d/99-security.conf
echo "* hard core 0" >> /etc/security/limits.conf
ulimit -c 0

# 5. Restrict /proc and /sys access
echo "[5] Hardening /proc..."
# Add to /etc/fstab:
# proc /proc proc nosuid,nodev,noexec,hidepid=2,gid=proc 0 0

# 6. Disable SysRq
echo "[6] Disabling SysRq..."
echo 0 > /proc/sys/kernel/sysrq
echo "kernel.sysrq = 0" >> /etc/sysctl.d/99-security.conf

# 7. Disable kernel module loading (after all modules are loaded)
echo "[7] Disabling module loading..."
# WARNING: Do this only after all required modules are loaded
# echo 1 > /proc/sys/kernel/modules_disabled

echo "=== Debug lockdown complete ==="
```

### 12.5 CoreSight Trace Control

The T234 includes ARM CoreSight debug and trace infrastructure. In production,
this must be restricted:

```bash
# CoreSight components on T234
ls /sys/bus/coresight/devices/
# Output includes: etm0, etm1, ..., etm5 (one per A78 core)
#                  funnel0, replicator0, tmc_etf, tmc_etr, stm

# Disable ETM (Embedded Trace Macrocell) on all cores
for etm in /sys/bus/coresight/devices/etm*; do
    echo 0 > "$etm/enable_source" 2>/dev/null
done

# Disable STM (System Trace Macrocell)
echo 0 > /sys/bus/coresight/devices/stm0/enable_source 2>/dev/null

# In production, CoreSight should be disabled via:
# 1. Kernel config: CONFIG_CORESIGHT=n
# 2. Device tree: remove or disable CoreSight nodes
# 3. Fuse: arm_jtag_disable (prevents external trace access)
```

---

## 13. Supply Chain Security

### 13.1 Board Identity Attestation

Every Jetson Orin Nano module has a unique identity anchored in hardware:

```
Device Identity Components:

+-------------------------------------------+
| Identity Layer     | Source               |
+--------------------+----------------------+
| Chip ECID          | T234 silicon fuse    |
|                    | (64-bit unique ID)   |
+--------------------+----------------------+
| Module serial no.  | EEPROM on module     |
+--------------------+----------------------+
| OEM device cert    | Provisioned at mfg   |
|                    | (X.509, ECC P-256)   |
+--------------------+----------------------+
| HUK-derived keys   | Fuse-derived,        |
|                    | unique per device    |
+--------------------+----------------------+
```

```bash
# Read the chip ECID (unique per-die identifier)
cat /sys/module/tegra_fuse/parameters/tegra_chip_uid
# Example: 0x0000000012345678

# Read module serial number from EEPROM
cat /proc/device-tree/serial-number
# or
i2cdump -y 0 0x50 b | head -20  # Module EEPROM at I2C addr 0x50

# The ECID is immutable and can serve as the root identity for
# device attestation and certificate binding.
```

### 13.2 Manufacturing Provisioning Workflow

```
Production Line Provisioning Flow:

Station 1: Board Assembly + ICT
  |
  v
Station 2: Flash Test Firmware (unsigned, JTAG enabled)
  |  - Basic board test (power rails, DRAM, peripherals)
  |  - Record ECID and module serial number
  |
  v
Station 3: Security Provisioning
  |  - Generate per-device key pair (ECC P-256)
  |  - Create CSR with ECID as subject
  |  - Submit CSR to OEM PKI CA
  |  - Receive signed device certificate
  |  - Generate per-device EKB (disk encryption key sealed to KEK2)
  |
  v
Station 4: Flash Signed Production Firmware
  |  - Flash signed bootloader, kernel, rootfs
  |  - Write device certificate to secure partition
  |  - Write EKB to EKB partition
  |  - Verify boot with signed firmware
  |
  v
Station 5: Burn Fuses
  |  - Burn PKC hash, SBK, KEK0/1/2
  |  - Burn security_mode fuse
  |  - Burn JTAG disable fuse
  |  - Verify fuse readback
  |
  v
Station 6: Final Verification
  |  - Cold boot test (must boot signed FW)
  |  - TLS attestation test (device proves identity to server)
  |  - Encrypted rootfs mount test
  |
  v
Station 7: Pack and Ship
```

### 13.3 Provisioning Server Implementation

```python
#!/usr/bin/env python3
"""
provision_device.py -- Per-device security provisioning script
Run on the manufacturing provisioning server.
"""

import subprocess
import json
import hashlib
import os
from pathlib import Path
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
import datetime

class DeviceProvisioner:
    def __init__(self, ca_cert_path, ca_key_path, ekb_template_dir):
        self.ca_cert_path = ca_cert_path
        self.ca_key_path = ca_key_path
        self.ekb_template_dir = ekb_template_dir

        # Load CA
        with open(ca_key_path, "rb") as f:
            self.ca_key = serialization.load_pem_private_key(f.read(), None)
        with open(ca_cert_path, "rb") as f:
            self.ca_cert = x509.load_pem_x509_certificate(f.read())

    def provision(self, ecid: str, module_serial: str) -> dict:
        """Provision a single device. Returns provisioning artifacts."""
        result = {}

        # 1. Generate per-device ECC P-256 key pair
        device_key = ec.generate_private_key(ec.SECP256R1())
        device_key_pem = device_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()
        )
        result["device_key"] = device_key_pem

        # 2. Create and sign device certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME,
                               f"jetson-{ecid}"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME,
                               "OEM Corporation"),
            x509.NameAttribute(NameOID.SERIAL_NUMBER,
                               module_serial),
        ])

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self.ca_cert.subject)
            .public_key(device_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(
                datetime.datetime.utcnow() + datetime.timedelta(days=3650)
            )
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .sign(self.ca_key, hashes.SHA256())
        )
        result["device_cert"] = cert.public_bytes(serialization.Encoding.PEM)

        # 3. Generate per-device disk encryption key
        disk_key = os.urandom(32)
        result["disk_key_hex"] = disk_key.hex()

        # 4. Generate EKB (sealed to KEK2)
        # In production, use NVIDIA's gen_ekb.py tool
        result["ekb_command"] = (
            f"python3 gen_ekb.py --chip 0x23 "
            f"--key {disk_key.hex()} "
            f"--fuse_key $KEK2_HEX "
            f"--out ekb_{ecid}.dat"
        )

        # 5. Record in provisioning database
        result["provision_record"] = {
            "ecid": ecid,
            "serial": module_serial,
            "cert_fingerprint": hashlib.sha256(
                result["device_cert"]
            ).hexdigest(),
            "provisioned_at": datetime.datetime.utcnow().isoformat(),
        }

        return result
```

### 13.4 Anti-Counterfeit Measures

```bash
# Runtime device authentication -- verify this is a genuine provisioned device

#!/bin/bash
# /usr/local/bin/device_attest.sh

ATTEST_SERVER="https://attest.example.com/verify"
DEVICE_CERT="/etc/ssl/certs/device.pem"
DEVICE_KEY="/etc/ssl/private/device.key"
CA_CERT="/etc/ssl/certs/fleet-ca.pem"

# 1. Request nonce from attestation server
NONCE=$(curl -s --cacert "$CA_CERT" "$ATTEST_SERVER/nonce")

# 2. Read device identity
ECID=$(cat /sys/module/tegra_fuse/parameters/tegra_chip_uid)

# 3. Create attestation claim
CLAIM=$(cat << EOF
{
    "ecid": "$ECID",
    "nonce": "$NONCE",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "fw_version": "$(cat /etc/nvidia/fw_version)",
    "secure_boot": "$(mokutil --sb-state 2>/dev/null || echo unknown)"
}
EOF
)

# 4. Sign the claim with device private key
echo "$CLAIM" | openssl dgst -sha256 -sign "$DEVICE_KEY" | base64 -w0 > /tmp/claim.sig

# 5. Send signed claim to attestation server
curl -s --cacert "$CA_CERT" \
    --cert "$DEVICE_CERT" --key "$DEVICE_KEY" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{\"claim\": $(echo "$CLAIM" | jq -c .), \"signature\": \"$(cat /tmp/claim.sig)\"}" \
    "$ATTEST_SERVER/attest"

rm -f /tmp/claim.sig
```

---

## 14. Vulnerability Management

### 14.1 CVE Monitoring for L4T/JetPack

```bash
# NVIDIA publishes security bulletins at:
# https://nvidia.custhelp.com/app/answers/detail/a_id/<id>
#
# Key components to monitor for CVEs:
#
# +---------------------------+-----------------------------------+
# | Component                 | CVE Sources                       |
# +---------------------------+-----------------------------------+
# | Linux Kernel (L4T)        | kernel.org, NVD, NVIDIA bulletins |
# | NVIDIA GPU driver         | NVIDIA security bulletins         |
# | UEFI firmware             | NVIDIA, Tianocore                 |
# | OP-TEE                    | github.com/OP-TEE advisories     |
# | ARM Trusted Firmware      | developer.arm.com                 |
# | OpenSSL / GnuTLS          | openssl.org, Ubuntu USNs          |
# | Container runtime         | Docker, NVIDIA container toolkit  |
# | TensorRT / CUDA           | NVIDIA security bulletins         |
# +---------------------------+-----------------------------------+

# Automated CVE scanning of the device
# Install vulnerability scanner
sudo apt-get install debsecan

# Scan for known vulnerabilities in installed packages
debsecan --suite jammy --only-fixed --format detail

# Scan container images for vulnerabilities
docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image nvcr.io/nvidia/l4t-tensorrt:r8.6
```

### 14.2 Security Update Process

```bash
#!/bin/bash
# /usr/local/bin/security_update.sh
# Automated security update workflow

set -euo pipefail

LOG="/var/log/security-updates.log"
LOCKFILE="/var/run/security-update.lock"

exec 200>"$LOCKFILE"
flock -n 200 || { echo "Update already in progress"; exit 1; }

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }

log "=== Security Update Check Started ==="

# 1. Update package lists
log "Updating package lists..."
apt-get update -qq 2>> "$LOG"

# 2. Check for security updates only
SECURITY_UPDATES=$(apt-get -s dist-upgrade 2>/dev/null | \
    grep -i "^Inst" | grep -i "security" | wc -l)
log "Security updates available: $SECURITY_UPDATES"

if [ "$SECURITY_UPDATES" -eq 0 ]; then
    log "No security updates needed."
    exit 0
fi

# 3. Download updates (but do not install yet)
log "Downloading security updates..."
apt-get -y --download-only dist-upgrade 2>> "$LOG"

# 4. Snapshot current state for rollback
log "Creating pre-update snapshot..."
SNAPSHOT_ID="pre-update-$(date +%Y%m%d%H%M%S)"
# If using A/B: updates go to inactive slot
# If using btrfs: btrfs subvolume snapshot
# If using overlayfs: save overlay state

# 5. Apply security updates
log "Applying security updates..."
DEBIAN_FRONTEND=noninteractive apt-get -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    dist-upgrade 2>> "$LOG"

# 6. Check if reboot is required
if [ -f /var/run/reboot-required ]; then
    log "Reboot required. Scheduling for maintenance window."
    # Schedule reboot for 3 AM local time
    echo "shutdown -r 03:00" | at now
else
    log "No reboot required."
fi

# 7. Report status to fleet management
curl -s --cert /etc/ssl/certs/device.pem \
    --key /etc/ssl/private/device.key \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{\"device\": \"$(hostname)\", \"updates_applied\": $SECURITY_UPDATES, \"reboot_needed\": $([ -f /var/run/reboot-required ] && echo true || echo false)}" \
    https://fleet-mgmt.example.com/api/v1/security-report 2>> "$LOG"

log "=== Security Update Check Complete ==="
```

### 14.3 Kernel Security Patching

```bash
# NVIDIA provides L4T kernel security patches via:
# 1. Full JetPack releases (major updates)
# 2. OTA packages via apt (minor security fixes)
# 3. Kernel source patches for custom kernels

# Check current kernel version and patch level
uname -r
# e.g., 5.15.136-tegra

# Check NVIDIA L4T version
cat /etc/nv_tegra_release
# or
dpkg -l nvidia-l4t-core | grep ii

# For custom kernels -- apply upstream security patches:
cd /path/to/kernel/source

# Apply a specific CVE fix
git log --oneline --grep="CVE-2024" upstream/linux-5.15.y
# Identify the relevant commit
git cherry-pick <commit-hash>

# Rebuild the kernel
make -j$(nproc) ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- Image modules dtbs

# Sign the new kernel for UEFI Secure Boot (see Section 3.7)
sbsign --key uefi_db.key --cert uefi_db.crt --output Image.signed Image
```

---

## 15. Security Monitoring

### 15.1 Audit Logging Configuration

```bash
# Install and configure auditd
sudo apt-get install auditd audispd-plugins

# /etc/audit/rules.d/jetson-security.rules
cat > /etc/audit/rules.d/jetson-security.rules << 'AUDITEOF'
## Delete all existing rules
-D

## Buffer size (increase for busy systems)
-b 8192

## Failure mode: 1 = printk, 2 = panic
-f 1

## Monitor authentication events
-w /etc/passwd -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/group -p wa -k identity
-w /etc/sudoers -p wa -k identity

## Monitor secure boot related files
-w /boot/ -p wa -k boot_integrity
-w /lib/optee_armtz/ -p wa -k optee_ta
-w /etc/nvidia/ -p wa -k nvidia_config

## Monitor firmware update events
-w /usr/local/bin/ota_verify.sh -p x -k ota_update
-w /usr/bin/nv_update_engine -p x -k firmware_update

## Monitor model files
-w /opt/models/ -p rwa -k model_access

## Monitor key material access
-w /etc/ssl/private/ -p rwa -k key_access
-w /data/tee/ -p rwa -k tee_storage

## Monitor privilege escalation attempts
-a always,exit -F arch=b64 -S execve -F euid=0 -F auid>=1000 -k privilege_escalation

## Monitor kernel module loading
-a always,exit -F arch=b64 -S init_module -S finit_module -k module_load
-a always,exit -F arch=b64 -S delete_module -k module_unload

## Monitor mount operations
-a always,exit -F arch=b64 -S mount -S umount2 -k mount_ops

## Monitor network configuration changes
-w /etc/nftables.conf -p wa -k firewall
-w /etc/wireguard/ -p wa -k vpn_config

## Monitor ptrace (anti-debug)
-a always,exit -F arch=b64 -S ptrace -k ptrace_access

## Make rules immutable (requires reboot to change)
-e 2
AUDITEOF

# Restart auditd
sudo systemctl restart auditd

# Search audit logs
ausearch -k model_access --start today
ausearch -k privilege_escalation -i
```

### 15.2 Intrusion Detection

```bash
# AIDE (Advanced Intrusion Detection Environment)
sudo apt-get install aide

# Configure AIDE for Jetson-specific paths
# /etc/aide/aide.conf.d/90_jetson
cat > /etc/aide/aide.conf.d/90_jetson << 'AIDEEOF'
# Monitor boot components
/boot        Full
/lib/firmware Full
/lib/modules  Full

# Monitor security configuration
/etc/nvidia   Full
/etc/ssl      Full
/etc/audit    Full
/etc/apparmor.d Full

# Monitor OP-TEE TAs
/lib/optee_armtz Full

# Monitor model files
/opt/models   Full

# Exclude frequently changing files
!/var/log
!/tmp
!/var/tmp
!/run
AIDEEOF

# Initialize AIDE database
sudo aideinit

# Run integrity check (typically via cron)
sudo aide --check

# Cron job for daily integrity checks
cat > /etc/cron.d/aide-check << 'CRONEOF'
0 2 * * * root /usr/bin/aide --check | /usr/local/bin/send_alert.sh
CRONEOF
```

### 15.3 Secure Remote Logging

```bash
# Configure rsyslog to send logs to a central SIEM over TLS

# /etc/rsyslog.d/50-remote-tls.conf
cat > /etc/rsyslog.d/50-remote-tls.conf << 'RSYSLOGEOF'
# Load TLS module
module(load="omrelp")

# TLS settings
global(
    DefaultNetstreamDriver="ossl"
    DefaultNetstreamDriverCAFile="/etc/ssl/certs/fleet-ca.pem"
    DefaultNetstreamDriverCertFile="/etc/ssl/certs/device.pem"
    DefaultNetstreamDriverKeyFile="/etc/ssl/private/device.key"
)

# Forward all security-relevant logs to central SIEM
if $syslogfacility-text == 'auth' or
   $syslogfacility-text == 'authpriv' or
   $syslogtag contains 'audit' or
   $msg contains 'SECURITY' then {
    action(
        type="omrelp"
        target="siem.example.com"
        port="2514"
        tls="on"
        tls.caCert="/etc/ssl/certs/fleet-ca.pem"
        tls.myCert="/etc/ssl/certs/device.pem"
        tls.myPrivKey="/etc/ssl/private/device.key"
        tls.authMode="name"
        tls.permittedpeer=["siem.example.com"]
        queue.type="LinkedList"
        queue.size="10000"
        queue.filename="siem_queue"
        queue.saveonshutdown="on"
        action.resumeRetryCount="-1"
    )
}
RSYSLOGEOF

sudo systemctl restart rsyslog
```

### 15.4 Tamper Detection

```bash
#!/bin/bash
# /usr/local/bin/tamper_detect.sh
# Run periodically to detect potential tampering

LOG="/var/log/tamper_detect.log"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }
alert() {
    log "ALERT: $*"
    logger -p auth.alert "TAMPER DETECTION: $*"
}

# 1. Check boot chain integrity
SECURE_BOOT=$(mokutil --sb-state 2>/dev/null || echo "unknown")
if echo "$SECURE_BOOT" | grep -qi "disabled"; then
    alert "Secure Boot is disabled"
fi

# 2. Check for unauthorized kernel modules
KNOWN_MODULES="/etc/nvidia/known_modules.list"
if [ -f "$KNOWN_MODULES" ]; then
    lsmod | awk 'NR>1{print $1}' | sort > /tmp/current_modules
    UNKNOWN=$(comm -13 "$KNOWN_MODULES" /tmp/current_modules)
    if [ -n "$UNKNOWN" ]; then
        alert "Unknown kernel modules loaded: $UNKNOWN"
    fi
    rm -f /tmp/current_modules
fi

# 3. Check for new SUID binaries
KNOWN_SUID="/etc/nvidia/known_suid.list"
if [ -f "$KNOWN_SUID" ]; then
    find / -perm -4000 -type f 2>/dev/null | sort > /tmp/current_suid
    NEW_SUID=$(comm -13 "$KNOWN_SUID" /tmp/current_suid)
    if [ -n "$NEW_SUID" ]; then
        alert "New SUID binaries detected: $NEW_SUID"
    fi
    rm -f /tmp/current_suid
fi

# 4. Check for unexpected open ports
EXPECTED_PORTS="22 "  # Only SSH expected inbound
OPEN_PORTS=$(ss -tlnp | awk 'NR>1{split($4,a,":"); print a[length(a)]}' | sort -u)
for port in $OPEN_PORTS; do
    if ! echo "$EXPECTED_PORTS" | grep -qw "$port"; then
        alert "Unexpected open port: $port"
    fi
done

# 5. Check OP-TEE TA integrity
for ta in /lib/optee_armtz/*.ta; do
    ta_name=$(basename "$ta")
    EXPECTED_HASH=$(grep "$ta_name" /etc/nvidia/ta_hashes.list 2>/dev/null | awk '{print $1}')
    if [ -n "$EXPECTED_HASH" ]; then
        ACTUAL_HASH=$(sha256sum "$ta" | awk '{print $1}')
        if [ "$EXPECTED_HASH" != "$ACTUAL_HASH" ]; then
            alert "TA integrity failure: $ta_name"
        fi
    fi
done

# 6. Check filesystem mount options
ROOT_OPTS=$(mount | grep "on / " | grep -o "([^)]*)")
if echo "$ROOT_OPTS" | grep -q "rw"; then
    log "INFO: Rootfs is mounted read-write"
    # For hardened production: rootfs should be read-only
fi

log "Tamper detection scan complete"
```

---

## 16. Production Hardening Checklist

### 16.1 Pre-Deployment Checklist

Use this checklist as the final verification before shipping a Jetson Orin Nano
product. Each item should be verified and signed off.

```
+----+--------------------------------------------------+--------+-------+
| #  | Item                                             | Status | Notes |
+----+--------------------------------------------------+--------+-------+
|    | HARDWARE SECURITY                                |        |       |
+----+--------------------------------------------------+--------+-------+
| 1  | PKC public key hash burned into fuses            | [ ]    |       |
| 2  | SBK (Secure Boot Key) burned into fuses          | [ ]    |       |
| 3  | KEK0, KEK1, KEK2 burned into fuses               | [ ]    |       |
| 4  | security_mode fuse burned                         | [ ]    |       |
| 5  | JTAG disable fuse burned                          | [ ]    |       |
| 6  | ARM JTAG disable fuse burned                      | [ ]    |       |
| 7  | ODM lock fuses burned (read-back protection)      | [ ]    |       |
| 8  | Fuse read-back verified (keys return zeros)        | [ ]    |       |
+----+--------------------------------------------------+--------+-------+
|    | BOOT SECURITY                                    |        |       |
+----+--------------------------------------------------+--------+-------+
| 9  | All bootloaders signed with OEM key               | [ ]    |       |
| 10 | Bootloader encryption with SBK enabled             | [ ]    |       |
| 11 | UEFI Secure Boot enabled and keys enrolled         | [ ]    |       |
| 12 | Linux kernel signed for UEFI Secure Boot           | [ ]    |       |
| 13 | Kernel module signature enforcement enabled         | [ ]    |       |
| 14 | dm-verity or rootfs read-only configured            | [ ]    |       |
| 15 | Anti-rollback counters initialized                  | [ ]    |       |
+----+--------------------------------------------------+--------+-------+
|    | DISK ENCRYPTION                                   |        |       |
+----+--------------------------------------------------+--------+-------+
| 16 | Root filesystem encrypted with LUKS2               | [ ]    |       |
| 17 | EKB provisioned per-device                         | [ ]    |       |
| 18 | Disk encryption key sealed via OP-TEE              | [ ]    |       |
| 19 | Data partition encrypted                            | [ ]    |       |
| 20 | Hardware AES acceleration verified                  | [ ]    |       |
+----+--------------------------------------------------+--------+-------+
|    | TRUSTED EXECUTION                                 |        |       |
+----+--------------------------------------------------+--------+-------+
| 21 | OP-TEE running and verified                        | [ ]    |       |
| 22 | tee-supplicant service enabled                     | [ ]    |       |
| 23 | hwkey-agent TA deployed                             | [ ]    |       |
| 24 | RPMB configured and authenticated                   | [ ]    |       |
| 25 | Custom TAs signed by OEM                            | [ ]    |       |
+----+--------------------------------------------------+--------+-------+
|    | NETWORK SECURITY                                  |        |       |
+----+--------------------------------------------------+--------+-------+
| 26 | Firewall rules configured (nftables/iptables)      | [ ]    |       |
| 27 | Only required ports open                            | [ ]    |       |
| 28 | TLS 1.2+ enforced for all connections               | [ ]    |       |
| 29 | mTLS configured for fleet management                | [ ]    |       |
| 30 | Per-device client certificates provisioned           | [ ]    |       |
| 31 | VPN configured for management plane                  | [ ]    |       |
| 32 | Unused network services disabled                     | [ ]    |       |
| 33 | TCP/IP stack hardened (sysctl)                       | [ ]    |       |
+----+--------------------------------------------------+--------+-------+
|    | DEBUG LOCKDOWN                                    |        |       |
+----+--------------------------------------------------+--------+-------+
| 34 | UART console disabled or password-protected         | [ ]    |       |
| 35 | Serial getty disabled                               | [ ]    |       |
| 36 | debugfs unmounted and disabled                      | [ ]    |       |
| 37 | kptr_restrict = 2                                   | [ ]    |       |
| 38 | dmesg_restrict = 1                                  | [ ]    |       |
| 39 | Core dumps disabled                                 | [ ]    |       |
| 40 | SysRq disabled                                      | [ ]    |       |
| 41 | CoreSight trace disabled                             | [ ]    |       |
+----+--------------------------------------------------+--------+-------+
|    | RUNTIME SECURITY                                  |        |       |
+----+--------------------------------------------------+--------+-------+
| 42 | AppArmor profiles for all services                  | [ ]    |       |
| 43 | seccomp profiles for inference workloads             | [ ]    |       |
| 44 | Capabilities dropped for all non-root services       | [ ]    |       |
| 45 | No-new-privileges flag set for all services           | [ ]    |       |
| 46 | Container security configured (if using Docker)       | [ ]    |       |
| 47 | User namespace remapping enabled                      | [ ]    |       |
+----+--------------------------------------------------+--------+-------+
|    | UPDATE SECURITY                                   |        |       |
+----+--------------------------------------------------+--------+-------+
| 48 | OTA update signing key provisioned                  | [ ]    |       |
| 49 | Update verification script deployed                  | [ ]    |       |
| 50 | A/B partition scheme configured                       | [ ]    |       |
| 51 | Anti-rollback mechanism active                        | [ ]    |       |
| 52 | Update channel uses TLS with pinned certificate       | [ ]    |       |
+----+--------------------------------------------------+--------+-------+
|    | MONITORING                                        |        |       |
+----+--------------------------------------------------+--------+-------+
| 53 | Audit logging configured (auditd)                   | [ ]    |       |
| 54 | Remote logging over TLS enabled                      | [ ]    |       |
| 55 | AIDE integrity monitoring initialized                | [ ]    |       |
| 56 | Tamper detection script deployed                      | [ ]    |       |
| 57 | Known-good baseline recorded                          | [ ]    |       |
+----+--------------------------------------------------+--------+-------+
|    | AI MODEL PROTECTION                               |        |       |
+----+--------------------------------------------------+--------+-------+
| 58 | Model files encrypted at rest                       | [ ]    |       |
| 59 | Model decryption via OP-TEE TA                       | [ ]    |       |
| 60 | Model integrity verification enabled                 | [ ]    |       |
| 61 | Model files on encrypted partition                    | [ ]    |       |
+----+--------------------------------------------------+--------+-------+
|    | IDENTITY AND ATTESTATION                          |        |       |
+----+--------------------------------------------------+--------+-------+
| 62 | Per-device X.509 certificate provisioned             | [ ]    |       |
| 63 | Device attestation endpoint tested                   | [ ]    |       |
| 64 | Device registered in fleet management system          | [ ]    |       |
+----+--------------------------------------------------+--------+-------+
```

### 16.2 Hardening Script

```bash
#!/bin/bash
# /usr/local/bin/production_harden.sh
# One-shot production hardening script for Jetson Orin Nano
# Run this AFTER all software is installed and tested

set -euo pipefail

echo "========================================"
echo " Jetson Orin Nano Production Hardening"
echo "========================================"
echo ""
echo "WARNING: This script applies irreversible changes."
echo "Ensure you have a tested recovery image available."
echo ""
read -p "Continue? [yes/NO] " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

LOG="/var/log/production_harden.log"
log() { echo "[$(date)] $*" | tee -a "$LOG"; }

# --- 1. User Account Hardening ---
log "=== User Account Hardening ==="

# Remove default user if present
if id "nvidia" &>/dev/null; then
    log "Removing default 'nvidia' user..."
    userdel -r nvidia 2>/dev/null || true
fi

# Set strong password policy
log "Setting password policy..."
cat >> /etc/security/pwquality.conf << 'EOF'
minlen = 12
dcredit = -1
ucredit = -1
lcredit = -1
ocredit = -1
maxrepeat = 3
EOF

# Lock root account (use sudo only)
log "Locking root account..."
passwd -l root

# Configure SSH key-only authentication
log "Hardening SSH..."
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#*X11Forwarding.*/X11Forwarding no/' /etc/ssh/sshd_config
sed -i 's/^#*AllowTcpForwarding.*/AllowTcpForwarding no/' /etc/ssh/sshd_config
echo "AllowGroups ssh-users" >> /etc/ssh/sshd_config

# --- 2. Filesystem Hardening ---
log "=== Filesystem Hardening ==="

# Set proper mount options
log "Hardening mount options in fstab..."
# Add noexec,nosuid,nodev to /tmp and /var/tmp
if ! grep -q "/tmp" /etc/fstab; then
    echo "tmpfs /tmp tmpfs noexec,nosuid,nodev,size=256M 0 0" >> /etc/fstab
fi

# Set sticky bit on world-writable directories
find / -xdev -type d \( -perm -0002 -a ! -perm -1000 \) -exec chmod o+t {} \; 2>/dev/null

# Remove world-writable files
find / -xdev -type f -perm -0002 -exec chmod o-w {} \; 2>/dev/null

# --- 3. Service Hardening ---
log "=== Service Hardening ==="

# Disable unnecessary services
DISABLE_SERVICES=(
    avahi-daemon
    cups
    cups-browsed
    bluetooth
    ModemManager
    whoopsie
    apport
)

for svc in "${DISABLE_SERVICES[@]}"; do
    if systemctl is-enabled "$svc" 2>/dev/null | grep -q enabled; then
        log "Disabling $svc..."
        systemctl disable "$svc"
        systemctl stop "$svc" 2>/dev/null || true
        systemctl mask "$svc"
    fi
done

# --- 4. Kernel Hardening ---
log "=== Kernel Hardening ==="

cat > /etc/sysctl.d/99-production-harden.conf << 'EOF'
# Kernel pointer hiding
kernel.kptr_restrict = 2

# Restrict dmesg
kernel.dmesg_restrict = 1

# Disable SysRq
kernel.sysrq = 0

# Restrict perf events
kernel.perf_event_paranoid = 3

# ASLR (full randomization)
kernel.randomize_va_space = 2

# Restrict core dumps
fs.suid_dumpable = 0

# Restrict access to kernel logs
kernel.printk = 3 3 3 3

# Restrict unprivileged user namespaces
kernel.unprivileged_userns_clone = 0

# Restrict eBPF
kernel.unprivileged_bpf_disabled = 1
net.core.bpf_jit_harden = 2

# Restrict ptrace
kernel.yama.ptrace_scope = 3
EOF

sysctl --system

# --- 5. Console Lockdown ---
log "=== Console Lockdown ==="

# Disable serial console
systemctl disable serial-getty@ttyTCU0.service 2>/dev/null || true
systemctl mask serial-getty@ttyTCU0.service 2>/dev/null || true

# Disable debugfs
if mountpoint -q /sys/kernel/debug; then
    umount /sys/kernel/debug
fi

# --- 6. Remove Development Tools ---
log "=== Removing Development Tools ==="

DEV_PACKAGES=(
    gdb
    strace
    ltrace
    tcpdump
    nmap
    netcat-openbsd
    gcc
    g++
    make
    python3-pip
)

for pkg in "${DEV_PACKAGES[@]}"; do
    if dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
        log "Removing $pkg..."
        apt-get -y remove --purge "$pkg" 2>/dev/null || true
    fi
done
apt-get -y autoremove

# --- 7. Record Known-Good Baseline ---
log "=== Recording Baseline ==="

# Record known kernel modules
lsmod | awk 'NR>1{print $1}' | sort > /etc/nvidia/known_modules.list

# Record known SUID binaries
find / -perm -4000 -type f 2>/dev/null | sort > /etc/nvidia/known_suid.list

# Record TA hashes
for ta in /lib/optee_armtz/*.ta; do
    sha256sum "$ta"
done > /etc/nvidia/ta_hashes.list

# Initialize AIDE
aideinit 2>/dev/null || true

log "========================================"
log " Production Hardening Complete"
log "========================================"
log " Reboot required to apply all changes."
log "========================================"
```

---

## 17. Common Issues and Debugging

### 17.1 Secure Boot Failures

**Symptom:** Device does not boot after fuse burning.

```
Common Causes and Solutions:

Problem: "MB1: BCT signature verification failed"
  Cause:  PKC hash fuse does not match the signing key used for BCT.
  Fix:    1. Verify the key used to sign matches the fused hash:
             python3 tegrasign_v3.py --pubkeyhash verify.bin --key oem.pem
             xxd verify.bin  # Compare with fused value
          2. Re-flash with correctly signed BCT (if JTAG still available).
          3. If JTAG is disabled: device is unrecoverable. This is why
             JTAG disable should be the LAST fuse burned.

Problem: "MB2: SBK decryption failed"
  Cause:  SBK fuse value does not match the encryption key used.
  Fix:    Verify the SBK used during signing matches the fused value.
          Use tegraflash with --encrypt_key pointing to correct SBK.

Problem: "UEFI Secure Boot violation"
  Cause:  Kernel image is not signed with an enrolled db key.
  Fix:    1. Boot into UEFI shell (if accessible).
          2. Enroll the correct db key.
          3. Or re-sign the kernel with the already-enrolled key.
```

```bash
# Debug secure boot failure from serial console (if UART is accessible)
# The BootROM prints error codes to UART at 115200 baud:

# Connect serial console
# Linux host:
screen /dev/ttyUSB0 115200

# Common BootROM error codes:
# 0x04 -- BCT read error (boot media issue)
# 0x14 -- BCT signature verification failed (PKC mismatch)
# 0x15 -- Bootloader decryption failed (SBK mismatch)
# 0x16 -- Bootloader signature verification failed
# 0x1F -- Security fuse check failed

# If device is stuck, force recovery mode:
# Hold REC button + press RESET to enter forced recovery mode
# Then re-flash with correctly signed images
```

### 17.2 Key Provisioning Errors

```bash
# Problem: EKB decryption fails during boot
# Symptom: "hwkey-agent: EKB authentication failed" in OP-TEE log

# Debug steps:
# 1. Verify EKB was generated with correct KEK2 value
echo "KEK2 used during EKB generation:"
xxd kek2.key

echo "KEK2 fuse value on device:"
# If fuse read-back is not locked yet:
cat /sys/devices/platform/efuse-burn/kek256
# If locked: the fuse cannot be read back, verify from provisioning records

# 2. Verify EKB binary integrity
sha256sum ekb.dat
# Compare with expected hash from provisioning system

# 3. Check OP-TEE logs for detailed error
# Enable OP-TEE debug logging (requires rebuild):
# In optee_os build: CFG_TEE_CORE_LOG_LEVEL=4
# Logs appear on secure UART (may be same as main UART)
dmesg | grep -i "optee\|tee\|hwkey"

# Problem: "tee-supplicant: failed to open /dev/tee0"
# Cause: OP-TEE driver not loaded or not probed
# Fix:
lsmod | grep optee
# If not loaded:
modprobe optee
# Check device tree for OP-TEE node:
dtc -I dtb -O dts /boot/dtb/*.dtb 2>/dev/null | grep -A5 "optee"
```

### 17.3 OP-TEE Communication Issues

```bash
# Problem: CA cannot open session with TA
# Error: "TEEC_OpenSession failed: 0xFFFF0008" (TEE_ERROR_TARGET_DEAD)

# Debug steps:

# 1. Verify TA is deployed
ls -la /lib/optee_armtz/
# TA files must be named exactly as their UUID with .ta extension
# Example: 8aaaf200-2450-11e4-abe2-0002a5d5c51b.ta

# 2. Check TA signature (if TA signing is enforced)
# OP-TEE can be configured to require signed TAs
# Check OP-TEE build config: CFG_REE_FS_TA_BUFFSIZE
# The TA signing key must match what OP-TEE was built with

# 3. Verify tee-supplicant is running and accessible
systemctl status tee-supplicant
ls -la /dev/tee0 /dev/teepriv0

# 4. Check permissions
# The user running the CA must be in the 'tee' group
id | grep tee
# If not:
sudo usermod -aG tee $(whoami)

# 5. Increase OP-TEE shared memory (if getting out-of-memory)
# In device tree:
#   optee {
#       compatible = "linaro,optee-tz";
#       method = "smc";
#       optee-os-shm-size = <0x400000>;  /* 4MB shared memory */
#   };

# Problem: "TEEC_InvokeCommand failed: 0xFFFF3024"
# This is TEE_ERROR_STORAGE_NO_SPACE
# The OP-TEE secure storage is full.
# Fix: Clean old secure storage objects or increase REE-FS allocation
du -sh /data/tee/
# Remove stale files (consult TA documentation before deleting)
```

### 17.4 Encryption Performance Impact

```bash
# Problem: Encrypted rootfs is causing slow boot and poor I/O performance

# Diagnose:
# 1. Measure baseline I/O without encryption
#    (on a test device before enabling LUKS)
fio --name=baseline --filename=/dev/mmcblk0p1 --rw=randread --bs=4k \
    --size=256M --numjobs=4 --direct=1 --group_reporting

# 2. Measure encrypted I/O
fio --name=encrypted --filename=/dev/mapper/jetson-rootfs --rw=randread \
    --bs=4k --size=256M --numjobs=4 --direct=1 --group_reporting

# 3. Check if hardware acceleration is active
cat /proc/crypto | grep -B3 "tegra"
# If no tegra driver is listed, software fallback is being used

# Fixes:
# A. Ensure tegra-se kernel module is loaded
modprobe tegra_se_aes
modprobe tegra_se_hash

# B. Verify dm-crypt is using the hardware driver
dmsetup table jetson-rootfs
# The cipher should show as "aes-xts-plain64"

# C. Tune dm-crypt work queues
# Disable workqueues (process crypto in submitting thread)
# This reduces latency for small I/O typical in inference workloads
cryptsetup --perf-no_read_workqueue --perf-no_write_workqueue \
    --allow-discards refresh jetson-rootfs

# D. Adjust I/O scheduler for encrypted device
echo "none" > /sys/block/dm-0/queue/scheduler

# E. Expected performance on Orin Nano 8GB with SE acceleration:
#
# +---------------------+----------+----------+-----------+
# | Workload            | No Enc   | SW Enc   | HW Enc    |
# +---------------------+----------+----------+-----------+
# | Sequential read     | 950 MB/s | 200 MB/s | 820 MB/s  |
# | Sequential write    | 480 MB/s | 180 MB/s | 450 MB/s  |
# | Random 4K read      | 28K IOPS | 12K IOPS | 25K IOPS  |
# | Random 4K write     | 22K IOPS | 10K IOPS | 20K IOPS  |
# | Boot time impact    | baseline | +8s      | +1.5s     |
# | TensorRT model load | 1.2s     | 3.8s     | 1.4s      |
# +---------------------+----------+----------+-----------+
```

### 17.5 Common Fuse Programming Mistakes

```
+------------------------------------------------------------------+
| Mistake                          | Consequence        | Recovery  |
+----------------------------------+--------------------+-----------+
| Burn security_mode before        | Device will not    | None if   |
| flashing signed firmware         | boot               | JTAG is   |
|                                  |                    | disabled  |
+----------------------------------+--------------------+-----------+
| Wrong PKC key hash fused         | Signed FW rejected | None --   |
|                                  | by BootROM         | bricked   |
+----------------------------------+--------------------+-----------+
| Disable JTAG before confirming   | Cannot debug boot  | None --   |
| signed boot works                | failures           | bricked   |
+----------------------------------+--------------------+-----------+
| Use wrong byte order for SBK     | Decryption fails   | None --   |
|                                  |                    | bricked   |
+----------------------------------+--------------------+-----------+
| Forget to backup fuse keys       | Cannot create new  | N/A --    |
| to HSM                           | signed firmware    | lost keys |
+----------------------------------+--------------------+-----------+
```

**Prevention checklist:**

```bash
# Before burning any fuses:
# 1. Backup all keys to HSM or air-gapped storage: YES / NO
# 2. Test signed firmware boots correctly:          YES / NO
# 3. Test signed firmware boots after reboot:       YES / NO
# 4. Test OTA update works with signed firmware:    YES / NO
# 5. Verify fuse config XML via --test mode:        YES / NO
# 6. Record device ECID in provisioning database:   YES / NO
# 7. At least 2 engineers review fuse config:       YES / NO
#
# ALL must be YES before proceeding with fuse burn.
```

### 17.6 Recovery Procedures

```bash
# Scenario: Device boots to signed FW but fails to mount encrypted rootfs

# 1. Boot into initrd recovery shell
# Add "break=premount" to kernel command line (requires UEFI shell access)

# 2. Manually unlock LUKS
cryptsetup luksOpen /dev/mmcblk0p1 rescue-rootfs
mount /dev/mapper/rescue-rootfs /mnt
chroot /mnt

# 3. Check and repair tee-supplicant / hwkey-agent
systemctl status tee-supplicant
ls -la /lib/optee_armtz/
ls -la /data/tee/

# Scenario: A/B update failed, device boots into old slot

# 1. Check which slot is active
nvbootctrl get-current-slot

# 2. Examine failed slot status
nvbootctrl dump-slots-info

# 3. If new slot failed validation, it was auto-rolled-back
# Check logs for the failure reason:
journalctl -b -1 | grep -i "boot\|update\|verify"

# 4. Mark failed slot as unbootable to prevent retry loops
nvbootctrl set-slot-as-unbootable B  # or A
```

### 17.7 Security Audit Verification

```bash
#!/bin/bash
# /usr/local/bin/security_audit.sh
# Run this script to verify the security posture of a production device

echo "========================================="
echo " Jetson Orin Nano Security Audit Report"
echo "========================================="
echo "Date: $(date -u)"
echo "Device: $(hostname)"
echo "ECID: $(cat /sys/module/tegra_fuse/parameters/tegra_chip_uid 2>/dev/null || echo N/A)"
echo ""

PASS=0
FAIL=0
WARN=0

check() {
    local desc="$1" result="$2"
    if [ "$result" = "PASS" ]; then
        echo "[PASS] $desc"
        PASS=$((PASS+1))
    elif [ "$result" = "FAIL" ]; then
        echo "[FAIL] $desc"
        FAIL=$((FAIL+1))
    else
        echo "[WARN] $desc"
        WARN=$((WARN+1))
    fi
}

# Boot security
SB=$(mokutil --sb-state 2>/dev/null)
if echo "$SB" | grep -qi "enabled"; then
    check "UEFI Secure Boot" "PASS"
else
    check "UEFI Secure Boot" "FAIL"
fi

# Kernel lockdown
LOCKDOWN=$(cat /sys/kernel/security/lockdown 2>/dev/null)
if echo "$LOCKDOWN" | grep -q "integrity\|confidentiality"; then
    check "Kernel lockdown" "PASS"
else
    check "Kernel lockdown" "WARN"
fi

# Disk encryption
if dmsetup status 2>/dev/null | grep -q "crypt"; then
    check "Disk encryption active" "PASS"
else
    check "Disk encryption active" "FAIL"
fi

# OP-TEE
if ls /dev/tee0 &>/dev/null; then
    check "OP-TEE available" "PASS"
else
    check "OP-TEE available" "FAIL"
fi

# tee-supplicant
if systemctl is-active tee-supplicant &>/dev/null; then
    check "tee-supplicant running" "PASS"
else
    check "tee-supplicant running" "FAIL"
fi

# Firewall
if nft list ruleset 2>/dev/null | grep -q "policy drop"; then
    check "Firewall default-drop policy" "PASS"
elif iptables -L INPUT 2>/dev/null | grep -q "DROP"; then
    check "Firewall default-drop policy" "PASS"
else
    check "Firewall default-drop policy" "FAIL"
fi

# SSH hardening
if grep -q "PasswordAuthentication no" /etc/ssh/sshd_config 2>/dev/null; then
    check "SSH password auth disabled" "PASS"
else
    check "SSH password auth disabled" "FAIL"
fi

# Debug interfaces
KPTR=$(cat /proc/sys/kernel/kptr_restrict 2>/dev/null)
if [ "$KPTR" = "2" ]; then
    check "Kernel pointer hiding" "PASS"
else
    check "Kernel pointer hiding" "FAIL"
fi

DMESG=$(cat /proc/sys/kernel/dmesg_restrict 2>/dev/null)
if [ "$DMESG" = "1" ]; then
    check "dmesg restricted" "PASS"
else
    check "dmesg restricted" "FAIL"
fi

SYSRQ=$(cat /proc/sys/kernel/sysrq 2>/dev/null)
if [ "$SYSRQ" = "0" ]; then
    check "SysRq disabled" "PASS"
else
    check "SysRq disabled" "WARN"
fi

# debugfs
if mountpoint -q /sys/kernel/debug 2>/dev/null; then
    check "debugfs unmounted" "FAIL"
else
    check "debugfs unmounted" "PASS"
fi

# Serial console
if systemctl is-enabled serial-getty@ttyTCU0.service 2>/dev/null | grep -q "enabled"; then
    check "Serial getty disabled" "FAIL"
else
    check "Serial getty disabled" "PASS"
fi

# AppArmor
AA_PROFILES=$(aa-status 2>/dev/null | grep "profiles are in enforce mode" | awk '{print $1}')
if [ "${AA_PROFILES:-0}" -gt 0 ]; then
    check "AppArmor enforcing ($AA_PROFILES profiles)" "PASS"
else
    check "AppArmor enforcing" "WARN"
fi

# Audit logging
if systemctl is-active auditd &>/dev/null; then
    check "Audit daemon running" "PASS"
else
    check "Audit daemon running" "WARN"
fi

# Default user removed
if id nvidia &>/dev/null; then
    check "Default 'nvidia' user removed" "FAIL"
else
    check "Default 'nvidia' user removed" "PASS"
fi

# Root account locked
ROOT_LOCKED=$(passwd -S root 2>/dev/null | awk '{print $2}')
if [ "$ROOT_LOCKED" = "L" ]; then
    check "Root account locked" "PASS"
else
    check "Root account locked" "WARN"
fi

# Core dumps disabled
COREDUMP=$(ulimit -c 2>/dev/null)
if [ "$COREDUMP" = "0" ]; then
    check "Core dumps disabled" "PASS"
else
    check "Core dumps disabled" "WARN"
fi

echo ""
echo "========================================="
echo " Audit Summary"
echo "========================================="
echo " PASS: $PASS"
echo " FAIL: $FAIL"
echo " WARN: $WARN"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then
    echo " RESULT: DEVICE IS NOT PRODUCTION READY"
    exit 1
else
    echo " RESULT: Device security posture is acceptable"
    exit 0
fi
```

---

## Appendix A: Quick Reference -- Key File Paths

```
+----------------------------------------------+-----------------------------------+
| Path                                         | Description                       |
+----------------------------------------------+-----------------------------------+
| /sys/devices/platform/efuse-burn/            | Fuse readback sysfs interface     |
| /sys/module/tegra_fuse/parameters/           | Tegra fuse parameters             |
| /dev/tee0                                    | OP-TEE normal world device        |
| /dev/teepriv0                                | OP-TEE privileged device          |
| /dev/hwrng                                   | Hardware RNG device               |
| /dev/tegra-se                                | Security Engine device (if avail) |
| /lib/optee_armtz/                            | OP-TEE Trusted Application store  |
| /data/tee/                                   | OP-TEE secure storage (encrypted) |
| /etc/nvidia/                                 | NVIDIA configuration files        |
| /boot/dtb/                                   | Device tree blobs                 |
| /boot/Image                                  | Linux kernel image                |
| /boot/extlinux/extlinux.conf                 | Boot configuration                |
| /proc/crypto                                 | Registered crypto algorithms      |
| /proc/sys/kernel/                            | Kernel security tunables          |
| /etc/nftables.conf                           | Firewall rules                    |
| /etc/audit/rules.d/                          | Audit rules                       |
| /etc/apparmor.d/                             | AppArmor profiles                 |
+----------------------------------------------+-----------------------------------+
```

## Appendix B: Quick Reference -- Key Commands

```bash
# Security Engine status
cat /proc/crypto | grep tegra

# OP-TEE version
dmesg | grep "optee: revision"

# UEFI Secure Boot status
mokutil --sb-state

# Fuse state
cat /sys/devices/platform/efuse-burn/security_mode

# Disk encryption status
dmsetup status

# Firewall rules
nft list ruleset

# AppArmor status
aa-status

# Audit log search
ausearch -k <key> --start today

# Device identity
cat /sys/module/tegra_fuse/parameters/tegra_chip_uid

# Boot slot info
nvbootctrl dump-slots-info
```

---

*End of Guide*
