// Compile-only on an x86 host: no target sysroot, allocation or C++ runtime required.
#include <arm_neon.h>
extern "C" float sum_four(float32x4_t value) { return vaddvq_f32(value); }
extern "C" int32x4_t truncate_four(float32x4_t value) { return vcvtq_s32_f32(value); }
