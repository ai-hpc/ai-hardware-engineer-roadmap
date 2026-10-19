#include "telemetry_frame.h"
#include <string.h>

uint16_t telemetry_crc16(const uint8_t *data, size_t size) {
    /* CRC-16/IBM-3740 (CCITT-FALSE): poly1021, initFFFF, no reflection/xorout. */
    uint16_t crc = UINT16_C(0xffff);
    for (size_t i = 0; i < size; ++i) {
        crc ^= (uint16_t)((uint16_t)data[i] << 8);
        for (unsigned bit = 0; bit < 8; ++bit) {
            crc = (uint16_t)((crc & UINT16_C(0x8000))
                             ? ((uint32_t)crc << 1) ^ UINT32_C(0x1021)
                             : (uint32_t)crc << 1);
        }
    }
    return crc;
}

size_t telemetry_encode(const telemetry_frame *frame, uint8_t *wire, size_t capacity) {
    if (!frame || !wire || frame->length > TELEMETRY_PAYLOAD_MAX ||
        capacity < TELEMETRY_WIRE_MAX) return 0;
    uint8_t raw[TELEMETRY_RAW_MAX];
    raw[0] = 1;  /* protocol version */
    raw[1] = (uint8_t)frame->sequence;
    raw[2] = (uint8_t)(frame->sequence >> 8);
    raw[3] = frame->length;
    memcpy(raw + 4, frame->payload, frame->length);
    const size_t body_size = 4u + frame->length;
    const uint16_t crc = telemetry_crc16(raw, body_size);
    raw[body_size] = (uint8_t)crc;
    raw[body_size + 1] = (uint8_t)(crc >> 8);
    /* Raw size <254, so each COBS run fits in one code byte. */
    size_t code_position = 0, next = 1;
    uint8_t code = 1;
    for (size_t i = 0; i < body_size + 2; ++i) {
        if (raw[i] == 0) {
            wire[code_position] = code;
            code_position = next++;
            code = 1;
        } else {
            wire[next++] = raw[i];
            ++code;
        }
    }
    wire[code_position] = code;
    wire[next++] = 0;  /* packet delimiter */
    return next;
}

void telemetry_parser_init(telemetry_parser *parser) {
    parser->used = 0;
    parser->discarding = false;
}

static telemetry_status decode(const uint8_t *encoded, size_t size,
                                telemetry_frame *output) {
    uint8_t raw[TELEMETRY_RAW_MAX];
    size_t input = 0, decoded = 0;
    while (input < size) {
        const uint8_t code = encoded[input++];
        if (code == 0 || (size_t)(code - 1u) > size - input)
            return TELEMETRY_BAD_COBS;
        for (unsigned i = 1; i < code; ++i) {
            if (decoded >= sizeof raw) return TELEMETRY_BAD_FORMAT;
            raw[decoded++] = encoded[input++];
        }
        if (code != UINT8_C(255) && input < size) {
            if (decoded >= sizeof raw) return TELEMETRY_BAD_FORMAT;
            raw[decoded++] = 0;
        }
    }
    if (decoded < 6 || raw[0] != 1 || raw[3] > TELEMETRY_PAYLOAD_MAX ||
        decoded != (size_t)raw[3] + 6u) return TELEMETRY_BAD_FORMAT;
    const uint16_t expected_crc = telemetry_crc16(raw, decoded - 2);
    const uint16_t actual_crc = (uint16_t)((uint16_t)raw[decoded - 2] |
                                          (uint16_t)((uint16_t)raw[decoded - 1] << 8));
    if (expected_crc != actual_crc) return TELEMETRY_BAD_CRC;
    telemetry_frame frame = {0};
    frame.sequence = (uint16_t)((uint16_t)raw[1] | (uint16_t)((uint16_t)raw[2] << 8));
    frame.length = raw[3];
    memcpy(frame.payload, raw + 4, frame.length);
    *output = frame;
    return TELEMETRY_READY;
}

telemetry_status telemetry_feed(telemetry_parser *parser, uint8_t byte,
                                telemetry_frame *output) {
    if (byte == 0) {
        if (parser->discarding || parser->used == 0) {
            telemetry_parser_init(parser);
            return TELEMETRY_NONE;
        }
        const size_t size = parser->used;
        parser->used = 0;
        return decode(parser->encoded, size, output);
    }
    if (parser->discarding) return TELEMETRY_NONE;
    if (parser->used == sizeof parser->encoded) {
        parser->used = 0;
        parser->discarding = true;
        return TELEMETRY_TOO_LONG;
    }
    parser->encoded[parser->used++] = byte;
    return TELEMETRY_NONE;
}

bool telemetry_sequence_newer(uint16_t candidate, uint16_t previous) {
    const uint16_t difference = (uint16_t)(candidate - previous);
    return difference != 0 && difference < UINT16_C(0x8000);
}

bool telemetry_age_valid(uint32_t now, uint32_t sampled, uint32_t maximum_age) {
    const uint32_t elapsed = now - sampled;
    return maximum_age < UINT32_C(0x80000000) &&
           elapsed < UINT32_C(0x80000000) && elapsed <= maximum_age;
}
