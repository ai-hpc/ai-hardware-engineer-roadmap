"""Small-object coordinate arithmetic, independent of any detector package."""
import json
import math


def letterbox(width, height, target_width=640, target_height=640):
    dims = (width, height, target_width, target_height)
    if any(not isinstance(x, int) or x <= 0 for x in dims):
        raise ValueError("positive integer image dimensions required")
    scale = min(target_width/width, target_height/height)
    # Record actual integer resize dimensions, then exact per-axis ratios.
    rw, rh = max(1, round(width*scale)), max(1, round(height*scale))
    return {"sx": rw/width, "sy": rh/height, "left": (target_width-rw)//2,
            "top": (target_height-rh)//2, "resized_width": rw, "resized_height": rh}


def encode_box(box, transform):
    x1, y1, x2, y2 = box
    return [x1*transform['sx']+transform['left'], y1*transform['sy']+transform['top'],
            x2*transform['sx']+transform['left'], y2*transform['sy']+transform['top']]


def decode_box(box, transform, crop_origin=(0, 0)):
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError("finite xyxy box required")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("positive box area required")
    ox, oy = crop_origin
    return [(x1-transform['left'])/transform['sx']+ox,
            (y1-transform['top'])/transform['sy']+oy,
            (x2-transform['left'])/transform['sx']+ox,
            (y2-transform['top'])/transform['sy']+oy]


def convert_visdrone(row, width, height):
    """Training-format convenience only; ignored regions need a separate policy."""
    if width <= 0 or height <= 0:
        raise ValueError("positive image dimensions required")
    values = [float(s.strip()) for s in row.strip().rstrip(',').split(',')]
    if len(values) != 8 or not all(math.isfinite(v) for v in values):
        raise ValueError("eight finite annotation fields required")
    x,y,w,h,score,category,truncation,occlusion = values
    if score not in (0,1) or category != int(category) or not 0 <= category <= 11:
        raise ValueError("expected ground-truth score 0/1 and category 0..11")
    if truncation not in (0,1) or occlusion not in (0,1,2) or w <= 0 or h <= 0:
        raise ValueError("invalid visibility fields or box dimensions")
    if score == 0 or category in (0,11):
        return {"label": None, "reason": "ignored", "raw_xywh": [x,y,w,h]}
    x1,y1,x2,y2 = max(0,x),max(0,y),min(width,x+w),min(height,y+h)
    if x2 <= x1 or y2 <= y1:
        return {"label": None, "reason": "outside_image"}
    return {"label": [int(category)-1,(x1+x2)/(2*width),(y1+y2)/(2*height),(x2-x1)/width,(y2-y1)/height],
            "reason": "retained", "was_clipped": (x1,y1,x2,y2)!=(x,y,x+w,y+h)}


def tile_origins(length, tile, overlap):
    if not all(isinstance(n,int) for n in (length,tile,overlap)) or length <= 0 or tile <= 0 or not 0 <= overlap < tile:
        raise ValueError("positive dimensions and overlap in [0,tile) required")
    if length <= tile:return [0]
    origins=list(range(0,length-tile+1,tile-overlap))
    if origins[-1] != length-tile:origins.append(length-tile)
    return origins


if __name__ == '__main__':
    full=letterbox(1920,1080); tile=letterbox(640,640)
    print(json.dumps({"full_frame_8px_object_width":8*full['sx'],"tile_8px_object_width":8*tile['sx'],
                      "full_frame_padding_top":full['top'],
                      "tile_origins_x":tile_origins(1920,640,128),"tile_origins_y":tile_origins(1080,640,128),
                      "sample_label":convert_visdrone('100,50,20,10,1,4,0,0',1920,1080)},indent=2))
