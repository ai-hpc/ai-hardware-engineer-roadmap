/* Build-only ABI fixture, using Linux headers. No CAN device is opened. */
#include <linux/can.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

_Static_assert(sizeof(struct can_frame) == 16, "unexpected Classic CAN ABI size");
_Static_assert(offsetof(struct can_frame, data) == 8, "unexpected data alignment");
_Static_assert(offsetof(struct can_frame, len) == 4, "unexpected length offset");

int main(int argc, char **argv) {
    const unsigned char payload[8] = {1, 127, 6, 255, 176, 4, 15, 0};
    struct can_frame frame = {0};
    if (argc == 2 && strcmp(argv[1], "emit") == 0) {
        frame.can_id = 0x321;
        frame.len = 8;
        memcpy(frame.data, payload, sizeof payload);
        return fwrite(&frame, sizeof frame, 1, stdout) == 1 ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "check") == 0) {
        if (fread(&frame, 1, sizeof frame, stdin) != sizeof frame || getchar() != EOF)
            return 2;
        if (frame.can_id != 0x321 || frame.len != 8 || memcmp(frame.data, payload, 8))
            return 3;
        puts("Linux can_frame ABI and independent payload fixture match");
        return 0;
    }
    fputs("usage: can_abi emit|check\n", stderr);
    return 64;
}
