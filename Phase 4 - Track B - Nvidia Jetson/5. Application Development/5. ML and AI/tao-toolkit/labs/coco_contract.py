"""Minimal COCO detection contract check; not a full COCO evaluator."""
from collections import Counter
import math


def validate(document):
    images, categories, annotations = (document[key] for key in ('images','categories','annotations'))
    for records in (images,categories,annotations):
        ids=[r['id'] for r in records]
        if any(not isinstance(i,int) or isinstance(i,bool) for i in ids) or len(ids)!=len(set(ids)):
            raise ValueError('unique integer IDs required in each collection')
    image_map={r['id']:r for r in images}
    class_ids={r['id'] for r in categories}
    if not image_map or not class_ids:
        raise ValueError('nonempty image and category metadata required')
    for r in images:
        dimensions = (r['width'], r['height'])
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in dimensions):
            raise ValueError('positive integer image dimensions required')
        if not isinstance(r['file_name'], str) or not r['file_name']:
            raise ValueError('nonempty file name required')
    counts=Counter()
    for a in annotations:
        if a['image_id'] not in image_map or a['category_id'] not in class_ids:
            raise ValueError('annotation references unknown image/category')
        box=a['bbox']
        if len(box)!=4 or not all(math.isfinite(x) for x in box):
            raise ValueError('finite xywh required')
        x,y,w,h=box;i=image_map[a['image_id']]
        if x<0 or y<0 or w<=0 or h<=0 or x+w>i['width'] or y+h>i['height']:
            raise ValueError('box outside image or nonpositive area')
        if a.get('iscrowd',0) not in (0,1):
            raise ValueError('invalid crowd flag')
        counts[a['category_id']]+=1
    return {'images':len(images),'annotations':len(annotations),'counts':dict(counts),
            'category_ids':sorted(class_ids),'images_without_annotations':len(set(image_map)-{a['image_id'] for a in annotations})}


def fixture():
    return {'images':[{'id':1,'width':100,'height':80,'file_name':'synthetic_a.png'},
                      {'id':2,'width':100,'height':80,'file_name':'synthetic_negative.png'}],
            'categories':[{'id':1,'name':'rectangle'},{'id':3,'name':'triangle'}],
            'annotations':[{'id':7,'image_id':1,'category_id':1,'bbox':[10,20,30,40],'iscrowd':0}]}


if __name__=='__main__':
    import json
    print(json.dumps(validate(fixture()),indent=2))
