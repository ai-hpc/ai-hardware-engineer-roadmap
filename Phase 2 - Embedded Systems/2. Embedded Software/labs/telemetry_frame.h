#ifndef COURSE_TELEMETRY_FRAME_H
#define COURSE_TELEMETRY_FRAME_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define TELEMETRY_PAYLOAD_MAX 32u
#define TELEMETRY_RAW_MAX (4u + TELEMETRY_PAYLOAD_MAX + 2u)
#define TELEMETRY_ENCODED_MAX (TELEMETRY_RAW_MAX + 1u)
#define TELEMETRY_WIRE_MAX (TELEMETRY_ENCODED_MAX + 1u)

typedef struct {
    uint16_t sequence;
    uint8_t length;
    uint8_t payload[TELEMETRY_PAYLOAD_MAX];
} telemetry_frame;

typedef struct {
    uint8_t encoded[TELEMETRY_ENCODED_MAX];
    size_t used;
    bool discarding;
} telemetry_parser;

typedef enum {
    TELEMETRY_NONE = 0,
    TELEMETRY_READY,
    TELEMETRY_BAD_COBS,
    TELEMETRY_BAD_FORMAT,
    TELEMETRY_BAD_CRC,
    TELEMETRY_TOO_LONG
} telemetry_status;

uint16_t telemetry_crc16(const uint8_t *data, size_t size);
/* Capacity must be at least TELEMETRY_WIRE_MAX. Returns 0 for an invalid frame. */
size_t telemetry_encode(const telemetry_frame *frame, uint8_t *wire, size_t capacity);
void telemetry_parser_init(telemetry_parser *parser);
/* Single-owner state machine. Output is modified only when READY is returned. */
telemetry_status telemetry_feed(telemetry_parser *parser, uint8_t byte,
                                telemetry_frame *output);
/* Same session; fewer than 32768 sequence positions may be ambiguous/outstanding. */
bool telemetry_sequence_newer(uint16_t candidate, uint16_t previous);
/* Same wrapping clock; comparisons require a window below half its uint32 range. */
bool telemetry_age_valid(uint32_t now, uint32_t sampled, uint32_t maximum_age);
#endif
