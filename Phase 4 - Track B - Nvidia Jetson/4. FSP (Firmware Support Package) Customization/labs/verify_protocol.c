#include "request_protocol.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "check failed line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

static void reply(struct course_message request, uint32_t value, uint8_t *wire) {
    request.kind = COURSE_REPLY; request.value = value;
    CHECK(course_encode(&request, wire, COURSE_FRAME_SIZE));
}
static void codec(void) {
    const struct course_message m = {COURSE_REPLY, UINT32_C(0x12345678), 42, UINT32_MAX};
    const uint8_t expected[16] = {1,2,0,0,0x12,0x34,0x56,0x78,0,0,0,42,255,255,255,255};
    uint8_t wire[18]; memset(wire, 0xA5, sizeof wire);
    CHECK(course_encode(&m, wire + 1, 16)); /* deliberately unaligned destination */
    CHECK(wire[0] == 0xA5 && wire[17] == 0xA5);
    CHECK(memcmp(wire + 1, expected, 16) == 0);
    struct course_message decoded = {0};
    CHECK(course_decode(wire + 1, 16, &decoded));
    CHECK(decoded.kind == m.kind && decoded.epoch == m.epoch &&
          decoded.sequence == m.sequence && decoded.value == m.value);
    for (size_t size = 0; size < 16; ++size) {
        decoded = (struct course_message){99,98,97,96};
        CHECK(!course_decode(wire + 1, size, &decoded));
        CHECK(decoded.kind == 99 && decoded.epoch == 98 && decoded.value == 96);
        uint8_t out[16]; memset(out, 0xCC, sizeof out);
        CHECK(!course_encode(&m, out, size));
        for (size_t i=0; i<sizeof out; ++i) CHECK(out[i] == 0xCC);
    }
    CHECK(!course_decode(wire + 1, 17, &decoded));
    const size_t corrupt[] = {0,1,2,3};
    for (size_t i=0; i<4; ++i) {
        memcpy(wire, expected, 16); wire[corrupt[i]] = 99;
        CHECK(!course_decode(wire, 16, &decoded));
    }
    memcpy(wire, expected, 16); memset(wire + 4, 0, 4);
    CHECK(!course_decode(wire,16,&decoded));
    memcpy(wire, expected, 16); memset(wire + 8, 0, 4);
    CHECK(!course_decode(wire,16,&decoded));
    CHECK(!course_decode(NULL,16,&decoded));
    CHECK(!course_decode(expected,16,NULL));
    CHECK(!course_encode(NULL,wire,16));
    CHECK(!course_encode(&m,NULL,16));
}
static void correlation(void) {
    struct course_client c = {0}; struct course_message req, req2;
    uint8_t wire[16]; uint32_t value=99;
    CHECK(course_init(&c,7)); CHECK(course_begin(&c,100,10,3,&req));
    CHECK(!course_begin(&c,100,10,4,&req2));
    struct course_message wrong=req; wrong.epoch=8; reply(wrong,42,wire);
    CHECK(course_receive(&c,101,wire,16,&value)==COURSE_REJECTED && c.pending && value==99);
    wrong=req; ++wrong.sequence; reply(wrong,42,wire);
    CHECK(course_receive(&c,101,wire,16,&value)==COURSE_REJECTED && c.pending);
    CHECK(course_encode(&req,wire,16));
    CHECK(course_receive(&c,101,wire,16,&value)==COURSE_REJECTED);
    reply(req,42,wire);
    CHECK(course_receive(&c,109,wire,16,&value)==COURSE_ACCEPTED && value==42 && !c.pending);
    CHECK(course_receive(&c,109,wire,16,&value)==COURSE_REJECTED); /* duplicate */
    CHECK(course_begin(&c,110,10,3,&req2)); CHECK(req2.sequence==req.sequence+1);
    CHECK(course_receive(&c,111,wire,16,&value)==COURSE_REJECTED && c.pending); /* stale */
    CHECK(course_init(&c,8)); CHECK(course_begin(&c,112,10,3,&req2));
    CHECK(req2.sequence==1); /* old reply has same sequence, different epoch */
    CHECK(course_receive(&c,113,wire,16,&value)==COURSE_REJECTED && c.pending);
    reply(req2,43,wire); CHECK(course_receive(&c,121,wire,16,&value)==COURSE_ACCEPTED && value==43);
}
static void time_boundaries(void) {
    const uint32_t starts[] = {0,1,UINT32_MAX-10,UINT32_MAX-1};
    const uint32_t timeouts[] = {1,2,10,1000,UINT32_C(0x7fffffff)};
    unsigned checked=0;
    for (size_t i=0;i<4;++i) for(size_t j=0;j<5;++j) {
        const uint64_t t=timeouts[j]; const uint64_t deltas[]={0,t-1,t,t+1,UINT64_C(0x80000000)};
        for(size_t k=0;k<5;++k) {
            struct course_client c; struct course_message req; uint8_t wire[16]; uint32_t value=999;
            CHECK(course_init(&c,1)); CHECK(course_begin(&c,starts[i],timeouts[j],0,&req));
            reply(req,123,wire);
            /* Reference uses an unwrapped 64-bit duration, independent of production comparison. */
            uint32_t now=(uint32_t)((uint64_t)starts[i]+deltas[k]);
            enum course_result expected=deltas[k]<t ? COURSE_ACCEPTED : COURSE_TIMED_OUT;
            CHECK(course_receive(&c,now,wire,16,&value)==expected);
            CHECK(!c.pending); CHECK(value==(expected==COURSE_ACCEPTED ? 123u : 999u));
            ++checked;
        }
    }
    CHECK(checked==100);
    struct course_client c; struct course_message req;
    CHECK(course_init(&c,1)); CHECK(course_begin(&c,UINT32_MAX-2,5,0,&req));
    CHECK(!course_poll(&c,1)); CHECK(course_poll(&c,2)); CHECK(!course_poll(&c,3));
    printf("100 deadline boundary cases including counter wrap passed\n");
}
static void limits(void) {
    struct course_client c={0}; struct course_message req={0};
    CHECK(!course_begin(&c,0,1,0,&req)); CHECK(!course_init(&c,0)); CHECK(!course_init(NULL,1));
    CHECK(course_init(&c,1));
    CHECK(!course_begin(&c,0,0,0,&req));
    CHECK(!course_begin(&c,0,UINT32_C(0x80000000),0,&req));
    CHECK(!course_begin(&c,0,UINT32_MAX,0,&req));
    CHECK(!course_begin(&c,0,1,0,NULL)); CHECK(c.last_sequence==0 && !c.pending);
    c.last_sequence=UINT32_MAX-1;
    CHECK(course_begin(&c,0,1,0,&req)); CHECK(req.sequence==UINT32_MAX);
    CHECK(course_poll(&c,1)); CHECK(!course_begin(&c,2,1,0,&req));
    CHECK(c.last_sequence==UINT32_MAX); /* no reused sequence in one epoch */
    CHECK(course_init(&c,2)); CHECK(course_begin(&c,2,1,0,&req)); CHECK(req.sequence==1);
    CHECK(!course_init(&c,0)); CHECK(c.pending && c.epoch==2);
}
int main(void) {
    codec(); correlation(); time_boundaries(); limits();
    puts("native protocol checks passed: codec, identity, deadlines, restart, exhaustion");
    return 0;
}
