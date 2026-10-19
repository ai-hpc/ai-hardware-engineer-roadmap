import unittest
from geometry import letterbox, encode_box, decode_box, convert_visdrone, tile_origins


class GeometryTests(unittest.TestCase):
    def test_letterbox_padding_and_round_trip(self):
        t=letterbox(1920,1080)
        self.assertEqual(t['top'],140)
        b=[100,50,108,58]
        decoded=decode_box(encode_box(b,t),t)
        for a,c in zip(b,decoded):self.assertAlmostEqual(a,c)
        self.assertAlmostEqual(8*t['sx'],8/3)

    def test_odd_image_rounding_and_crop_offset(self):
        t=letterbox(853,479)
        b=[1,2,10,12]
        got=decode_box(encode_box(b,t),t,(512,0))
        for a,c in zip(got,[513,2,522,12]):self.assertAlmostEqual(a,c)

    def test_ignored_and_clipped_annotations(self):
        self.assertEqual(convert_visdrone('1,2,3,4,0,4,0,0',100,100)['reason'],'ignored')
        self.assertEqual(convert_visdrone('1,2,3,4,1,11,0,0',100,100)['reason'],'ignored')
        r=convert_visdrone('-5,10,10,10,1,4,0,0',100,100)
        self.assertEqual(r['label'],[3,.025,.15,.05,.1])
        self.assertTrue(r['was_clipped'])
        for row in ('0,0,1,1,1,1.5,0,0','0,0,-1,1,1,1,0,0','nan,0,1,1,1,1,0,0'):
            with self.assertRaises(ValueError):convert_visdrone(row,100,100)

    def test_edge_tiles_cover_image(self):
        self.assertEqual(tile_origins(1920,640,128),[0,512,1024,1280])
        self.assertEqual(tile_origins(1080,640,128),[0,440])
        self.assertEqual(tile_origins(200,640,128),[0])
        with self.assertRaises(ValueError):tile_origins(1920,640,640)


if __name__=='__main__':unittest.main()
