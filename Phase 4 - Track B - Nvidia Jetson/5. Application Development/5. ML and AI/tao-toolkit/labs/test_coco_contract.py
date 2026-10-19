import unittest
from coco_contract import validate, fixture


class COCOTests(unittest.TestCase):
    def test_sparse_ids_and_negative_image_are_explicit(self):
        result=validate(fixture())
        self.assertEqual(result['category_ids'],[1,3])
        self.assertEqual(result['images_without_annotations'],1)

    def test_unknown_reference_duplicate_ids_and_bad_geometry(self):
        d=fixture();d['annotations'][0]['category_id']=2
        with self.assertRaises(ValueError):validate(d)
        d=fixture();d['annotations'].append(dict(d['annotations'][0]))
        with self.assertRaises(ValueError):validate(d)
        for box in ([10,20,100,20],[1,1,0,2],[float('nan'),1,2,3]):
            d=fixture();d['annotations'][0]['bbox']=box
            with self.assertRaises(ValueError):validate(d)
        for width in (0, -1, True, float('nan'), 12.5):
            d=fixture();d['images'][0]['width']=width
            with self.assertRaises(ValueError):validate(d)


if __name__=='__main__':unittest.main()
