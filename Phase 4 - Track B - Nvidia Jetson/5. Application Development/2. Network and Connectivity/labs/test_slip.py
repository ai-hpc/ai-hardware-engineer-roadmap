import unittest
from slip import Decoder, encode


class SlipTests(unittest.TestCase):
    def test_independent_wire_bytes_and_every_byte_fragmented(self):
        self.assertEqual(encode(bytes([0x01,0xC0,0xDB])),bytes([0xC0,0x01,0xDB,0xDC,0xDB,0xDD,0xC0]))
        decoder=Decoder();frames=[]
        for b in encode(bytes(range(256))):frames.extend(decoder.feed(bytes([b])))
        self.assertEqual(frames,[bytes(range(256))])

    def test_coalesced_frames_ignore_empty_boundaries(self):
        self.assertEqual(Decoder().feed(encode(b'a')+encode(b'b')),[b'a',b'b'])

    def test_invalid_escape_and_oversize_resynchronize(self):
        decoder=Decoder(maximum=3)
        self.assertEqual(decoder.feed(b'\xc0\xdb\x11\x20\xc0'+encode(b'1234')+encode(b'ok')),[b'ok'])
        self.assertEqual(decoder.rejected,2)
        self.assertLessEqual(len(decoder.buffer),3)
        self.assertEqual(decoder.feed(b'\xdb\xc0'+encode(b'yes')),[b'yes'])
        self.assertEqual(decoder.rejected,3)


if __name__=='__main__':unittest.main()
