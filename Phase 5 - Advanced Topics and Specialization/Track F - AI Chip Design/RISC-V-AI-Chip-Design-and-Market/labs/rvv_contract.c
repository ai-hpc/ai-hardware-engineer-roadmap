// Freestanding Linux/RISC-V executable: no libc, board SDK or vendor extensions.
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

static int32_t a[160], b[160], output[160];
static const int32_t poison = INT32_MIN;

static long write_stdout(const char *text, size_t length) {
    register long a0 __asm__("a0") = 1;
    register long a1 __asm__("a1") = (long)text;
    register long a2 __asm__("a2") = (long)length;
    register long a7 __asm__("a7") = 64;  // Linux write on RISC-V
    __asm__ volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
    return a0;
}

__attribute__((noreturn)) static void exit_process(int code) {
    register long a0 __asm__("a0") = code;
    register long a7 __asm__("a7") = 93;  // Linux exit
    __asm__ volatile("ecall" : : "r"(a0), "r"(a7) : "memory");
    __builtin_unreachable();
}

__attribute__((noinline)) static size_t vector_transform(
        const int32_t *left, const int32_t *right, int32_t *result, size_t n) {
    size_t offset = 0, processed = 0;
    while (offset < n) {
#ifdef COURSE_MUTANT_TAIL
        const size_t vl = __riscv_vsetvlmax_e32m1();
#else
        const size_t vl = __riscv_vsetvl_e32m1(n - offset);
#endif
        const vint32m1_t va = __riscv_vle32_v_i32m1(left + offset, vl);
        const vint32m1_t vb = __riscv_vle32_v_i32m1(right + offset, vl);
        const vint32m1_t doubled = __riscv_vadd_vv_i32m1(va, va, vl);
        const vint32m1_t value = __riscv_vadd_vv_i32m1(doubled, vb, vl);
        __riscv_vse32_v_i32m1(result + offset, value, vl);
        processed += vl;
#ifdef COURSE_MUTANT_FIXED_WIDTH
        offset += 8;  // Deliberately wrong except when the useful width happens to be eight.
#else
        offset += vl;
#endif
    }
    return processed;
}

static int check_all(void) {
    for (size_t n = 0; n <= 97; ++n) {
        for (size_t shift = 0; shift < 8; ++shift) {
            const size_t begin = 16 + shift;
            for (size_t i = 0; i < 160; ++i) {
                a[i] = (int32_t)(i % 17) - 8;
                b[i] = (int32_t)(i % 11) - 5;
                output[i] = poison;
            }
            const size_t count = vector_transform(a + begin, b + begin, output + begin, n);
            if (count != n) return 1;
            for (size_t i = 0; i < 160; ++i) {
                if (i >= begin && i < begin + n) {
                    // Small bounded operands avoid signed overflow in this scalar reference.
                    const int32_t expected = 2 * ((int32_t)(i % 17) - 8) + ((int32_t)(i % 11) - 5);
                    if (output[i] != expected) return 2;
                } else if (output[i] != poison) return 3;
            }
        }
    }
    return 0;
}

__attribute__((noreturn)) void _start(void) {
    const int result = check_all();
    const char pass[] = "PASS: 784 length/offset cases; guards and processed count agree\n";
    const char fail[] = "FAIL: vector width, tail, or value contract\n";
    const char *message = result ? fail : pass;
    size_t remaining = result ? sizeof(fail)-1 : sizeof(pass)-1;
    while (remaining) {
        const long written = write_stdout(message, remaining);
        if (written <= 0) exit_process(4);
        message += written;
        remaining -= (size_t)written;
    }
    exit_process(result);
}
