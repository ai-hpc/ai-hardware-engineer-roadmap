#include "request_protocol.h"

static bool valid(const struct course_message *m) {
    return m && (m->kind == COURSE_REQUEST || m->kind == COURSE_REPLY) &&
           m->epoch != 0 && m->sequence != 0;
}
static uint32_t read32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}
static void write32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v;
}
bool course_encode(const struct course_message *m, uint8_t *out, size_t capacity) {
    if (!valid(m) || !out || capacity < COURSE_FRAME_SIZE) return false;
    out[0] = 1; out[1] = m->kind; out[2] = 0; out[3] = 0;
    write32(out + 4, m->epoch); write32(out + 8, m->sequence);
    write32(out + 12, m->value);
    return true;
}
bool course_decode(const uint8_t *wire, size_t length, struct course_message *out) {
    if (!wire || !out || length != COURSE_FRAME_SIZE || wire[0] != 1 ||
        wire[2] != 0 || wire[3] != 0) return false;
    const struct course_message m = {wire[1], read32(wire + 4),
                                    read32(wire + 8), read32(wire + 12)};
    if (!valid(&m)) return false;
    *out = m;
    return true;
}
bool course_init(struct course_client *c, uint32_t epoch) {
    if (!c || epoch == 0) return false;
    *c = (struct course_client){.epoch = epoch};
    return true;
}
bool course_begin(struct course_client *c, uint32_t now, uint32_t timeout,
                  uint32_t selector, struct course_message *out) {
    if (!c || !out || c->epoch == 0 || c->pending || timeout == 0 ||
        timeout >= UINT32_C(0x80000000) || c->last_sequence == UINT32_MAX) return false;
    ++c->last_sequence;
    c->started = now; c->timeout = timeout; c->pending = true;
    *out = (struct course_message){COURSE_REQUEST, c->epoch, c->last_sequence, selector};
    return true;
}
bool course_poll(struct course_client *c, uint32_t now) {
    if (!c || !c->pending) return false;
    if ((uint32_t)(now - c->started) < c->timeout) return false;
    c->pending = false;
    return true;
}
enum course_result course_receive(struct course_client *c, uint32_t now,
                                  const uint8_t *wire, size_t length, uint32_t *value) {
    struct course_message m;
    if (!c || !value || !c->pending || !course_decode(wire, length, &m) ||
        m.kind != COURSE_REPLY || m.epoch != c->epoch ||
        m.sequence != c->last_sequence) return COURSE_REJECTED;
    if (course_poll(c, now)) return COURSE_TIMED_OUT;
    *value = m.value;
    c->pending = false;
    return COURSE_ACCEPTED;
}
