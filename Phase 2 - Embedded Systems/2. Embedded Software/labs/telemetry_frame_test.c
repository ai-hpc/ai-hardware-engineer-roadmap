#include "telemetry_frame.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line%d: %s\n", __LINE__, #condition); exit(1); \
} } while (0)

static telemetry_status consume(telemetry_parser *parser, const uint8_t *wire,
                                size_t size, telemetry_frame *output) {
    telemetry_status result = TELEMETRY_NONE;
    for (size_t i = 0; i < size; ++i) {
        telemetry_status current = telemetry_feed(parser, wire[i], output);
        if (current != TELEMETRY_NONE) result = current;
    }
    return result;
}

static void boundaries(void) {
    const uint8_t known[] = "123456789";
    REQUIRE(telemetry_crc16(known, 9) == UINT16_C(0x29b1));
    REQUIRE(telemetry_crc16(known, 0) == UINT16_C(0xffff));
    for (unsigned length = 0; length <= TELEMETRY_PAYLOAD_MAX; ++length) {
        for (unsigned pattern = 0; pattern < 3; ++pattern) {
            telemetry_frame input = {.sequence = UINT16_C(65535), .length = (uint8_t)length};
            for (unsigned i = 0; i < length; ++i)
                input.payload[i] = pattern == 0 ? 0 : pattern == 1 ? 255 : (uint8_t)(i * 17u);
            uint8_t wire[TELEMETRY_WIRE_MAX + 2];
            memset(wire, 0x6b, sizeof wire);
            size_t size = telemetry_encode(&input, wire, TELEMETRY_WIRE_MAX);
            REQUIRE(size == length + 8u);
            REQUIRE(wire[size - 1] == 0);
            REQUIRE(wire[TELEMETRY_WIRE_MAX] == 0x6b);
            REQUIRE(wire[TELEMETRY_WIRE_MAX + 1] == 0x6b);
            for (size_t i = 0; i + 1 < size; ++i) REQUIRE(wire[i] != 0);
            /* Every possible read boundary: UART reads need not align to packets. */
            for (size_t split = 0; split < size; ++split) {
                telemetry_parser parser;
                telemetry_frame output = {0};
                telemetry_parser_init(&parser);
                REQUIRE(consume(&parser, wire, split, &output) == TELEMETRY_NONE);
                REQUIRE(consume(&parser, wire + split, size - split, &output) == TELEMETRY_READY);
                REQUIRE(output.sequence == input.sequence && output.length == input.length);
                REQUIRE(memcmp(output.payload, input.payload, length) == 0);
                /* Consecutive frame without resetting the parser. */
                REQUIRE(consume(&parser, wire, size, &output) == TELEMETRY_READY);
            }
        }
    }
}

static void rejection_and_recovery(void) {
    telemetry_parser parser;
    telemetry_parser_init(&parser);
    telemetry_frame input = {.sequence = 7, .length = 3, .payload = {1, 2, 3}};
    telemetry_frame output = {.sequence = 999, .length = 1, .payload = {77}};
    uint8_t valid[TELEMETRY_WIRE_MAX], damaged[TELEMETRY_WIRE_MAX];
    const size_t size = telemetry_encode(&input, valid, sizeof valid);
    REQUIRE(size != 0);
    /* Last encoded byte is a nonzero CRC byte for this fixture. */
    memcpy(damaged, valid, size);
    damaged[size - 2] ^= 1;
    REQUIRE(damaged[size - 2] != 0);
    REQUIRE(consume(&parser, damaged, size, &output) == TELEMETRY_BAD_CRC);
    REQUIRE(output.sequence == 999 && output.payload[0] == 77);
    REQUIRE(consume(&parser, valid, size, &output) == TELEMETRY_READY);
    /* Version is the first raw byte and first encoded data byte for this fixture. */
    memcpy(damaged, valid, size);
    damaged[1] = 2;
    REQUIRE(consume(&parser, damaged, size, &output) == TELEMETRY_BAD_FORMAT);
    REQUIRE(consume(&parser, valid, size, &output) == TELEMETRY_READY);
    const uint8_t malformed[] = {5, 1, 0};
    REQUIRE(consume(&parser, malformed, sizeof malformed, &output) == TELEMETRY_BAD_COBS);
    REQUIRE(consume(&parser, valid, size, &output) == TELEMETRY_READY);
    /* Truncated frame cannot yield READY; delimiter rejects it and resynchronizes. */
    REQUIRE(consume(&parser, valid, size - 3, &output) == TELEMETRY_NONE);
    REQUIRE(telemetry_feed(&parser, 0, &output) != TELEMETRY_READY);
    REQUIRE(consume(&parser, valid, size, &output) == TELEMETRY_READY);
    for (size_t i = 0; i < TELEMETRY_ENCODED_MAX; ++i)
        REQUIRE(telemetry_feed(&parser, 1, &output) == TELEMETRY_NONE);
    REQUIRE(telemetry_feed(&parser, 1, &output) == TELEMETRY_TOO_LONG);
    for (unsigned i = 0; i < 1000; ++i)
        REQUIRE(telemetry_feed(&parser, 1, &output) == TELEMETRY_NONE);
    REQUIRE(parser.used == 0 && parser.discarding);
    REQUIRE(telemetry_feed(&parser, 0, &output) == TELEMETRY_NONE);
    REQUIRE(consume(&parser, valid, size, &output) == TELEMETRY_READY);
    REQUIRE(telemetry_feed(&parser, 0, &output) == TELEMETRY_NONE);
    input.length = TELEMETRY_PAYLOAD_MAX + 1;
    REQUIRE(telemetry_encode(&input, valid, sizeof valid) == 0);
    input.length = 0;
    REQUIRE(telemetry_encode(&input, valid, TELEMETRY_WIRE_MAX - 1) == 0);
    REQUIRE(telemetry_encode(NULL, valid, sizeof valid) == 0);
    REQUIRE(telemetry_encode(&input, NULL, sizeof valid) == 0);
}

static void wrapping_counters(void) {
    REQUIRE(telemetry_sequence_newer(0, 65535));
    REQUIRE(telemetry_sequence_newer(32767, 0));
    REQUIRE(!telemetry_sequence_newer(32768, 0));
    REQUIRE(!telemetry_sequence_newer(7, 7));
    REQUIRE(!telemetry_sequence_newer(6, 7));
    REQUIRE(telemetry_age_valid(3, UINT32_MAX - 2, 6));
    REQUIRE(!telemetry_age_valid(3, UINT32_MAX - 2, 5));
    REQUIRE(telemetry_age_valid(100, 100, 0));
    REQUIRE(!telemetry_age_valid(99, 100, 20));
    REQUIRE(!telemetry_age_valid(UINT32_C(0x80000000), 0, UINT32_C(0x7fffffff)));
    REQUIRE(!telemetry_age_valid(100, 100, UINT32_C(0x80000000)));
}

int main(void) {
    boundaries();
    rejection_and_recovery();
    wrapping_counters();
    puts("PASS: CRC vector, 99 payload fixtures/all byte splits, rejection/recovery, counter wrap");
    return 0;
}
