import numpy as np
import json
import shutil
import os


def catFinder(cat_ids, coco_json_path, new_ann_path):
    # read the data
    f = open(coco_json_path)
    data = json.load(f)
    
    new_annotations = {}
    # find images with certain categories
    for ann in data['annotations']:
        if ann['category_id'] in cat_ids:
            image_id = str(ann['image_id'])        
            category_id = ann['category_id']
            bbox = ann['bbox']
            
            try:
                new_annotations[image_id][0]['category_id'].append(category_id)
                new_annotations[image_id][0]['bbox'].append(bbox)
            except KeyError:
                new_annotations[image_id] = []
                new_annotations[image_id].append({'category_id': [category_id], 'bbox': [bbox]})
                
    # find the corosponding file names
    for key in new_annotations.keys():
        for img in data['images']:
            if key == str(img['id']):
                new_annotations[key][0]['file_name'] = img['file_name']
                new_annotations[key][0]['height'] = img['height']
                new_annotations[key][0]['width'] = img['width']
    
    json_file = json.dumps(new_annotations)
    f = open(new_ann_path,"w")
    f.write(json_file)
    
    print('Found all images with your categories :)')
        
    return new_annotations


def copyImgs(new_ann_path, new_img_path):    
    try:
        os.mkdir(new_img_path)
    except FileExistsError:
        pass
    
    f = open(new_ann_path)
    data = json.load(f)
    for e in data.values():
        file_name = e[0]['file_name']
        shutil.copy(os.path.join('train2017', file_name), os.path.join(new_img_path, file_name))
    
    print('Copied all images with those categories to the new directory')


def toYOLO(data, yolo_ann_path):
    try:
        os.mkdir(yolo_ann_path)
    except FileExistsError:
        pass
    
    for e in data.values():
        e = e[0]
        for i, bbox in enumerate(e['bbox']):
            xc = (bbox[0]+bbox[2])/e['width']
            yc = (bbox[1]+bbox[3])/e['height']
            width = bbox[2]/e['width']
            height = bbox[3]/e['height']
            
            e['bbox'][i] = [e['category_id'][i], xc, yc, width, height]
            
        np.savetxt(os.path.join(yolo_ann_path,e['file_name'][:-4]+'.txt'), e['bbox'], fmt='%.10g')

    
    print('All annotations converted to YOLO format')
    print('Done!')


# [bird,cat,dog, backpack,umbrella,handbag,suitcase, chair,couch,dining table, ,laptop,cellphone,book]
cat_ids = [2,3,4,6,8, 16,17,18, 27,28,31,33, 62,63,69, 73,77, 84] 
new_ann_path = 'annotations/new_instances_train2017.json'
coco_json_path = 'annotations/instances_train2017.json'

data = catFinder(cat_ids, coco_json_path,  new_ann_path)
        
new_img_path = 'new_train2017'
copyImgs(new_ann_path, new_img_path)    

yolo_ann_path = 'yolo'
toYOLO(data, yolo_ann_path)