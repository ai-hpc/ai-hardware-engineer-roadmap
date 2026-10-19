#ifndef COURSE_REQUEST_PROTOCOL_H
#define COURSE_REQUEST_PROTOCOL_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Teaching wire format, not NVIDIA's IVC framing or firmware ABI.
 * One serialized caller owns each client; no locks/cache/MMIO operations here.
 * now uses a wrapping uint32 clock; service before 2^31 ticks have elapsed.
 * timeout is 1..2^31-1 ticks. Epoch must not be reused while old frames exist.
 */
#define COURSE_FRAME_SIZE 16u
#define COURSE_REQUEST 1u
#define COURSE_REPLY 2u
struct course_message { uint8_t kind; uint32_t epoch, sequence, value; };
struct course_client {
    uint32_t epoch, last_sequence, started, timeout;
    bool pending;
};
enum course_result { COURSE_REJECTED, COURSE_ACCEPTED, COURSE_TIMED_OUT };

bool course_encode(const struct course_message *message, uint8_t *out, size_t capacity);
bool course_decode(const uint8_t *wire, size_t length, struct course_message *out);
/* init/new session explicitly discards a pending correlation, not device work. */
bool course_init(struct course_client *client, uint32_t epoch);
bool course_begin(struct course_client *client, uint32_t now, uint32_t timeout,
                  uint32_t selector, struct course_message *out);
bool course_poll(struct course_client *client, uint32_t now);
enum course_result course_receive(struct course_client *client, uint32_t now,
                                  const uint8_t *wire, size_t length, uint32_t *value);
#endif
