import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def dataset_loader(json_path, image_root, thing_classes):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Build maps
    id_to_img = {img["id"]: img for img in coco_data["images"]}
    id_to_cat = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Organize annotations by image
    image_to_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)

    dataset_dicts = []
    for img_id, anns in image_to_anns.items():
        img = id_to_img[img_id]
        record = {
            "file_name": os.path.join(image_root, img["file_name"]),
            "image_id": img_id,
            "height": img["height"],
            "width": img["width"],
            "annotations": [],
        }

        for ann in anns:
            category_name = id_to_cat[ann["category_id"]]
            if category_name not in thing_classes:
                continue  # skip if not in defined class list

            record["annotations"].append({
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": thing_classes.index(category_name),
            })

        dataset_dicts.append(record)

    return dataset_dicts


def register_cigbutt_dataset(name, json_path, image_root, thing_classes, metadata):
    DatasetCatalog.register(
        name,
        lambda: dataset_loader(json_path, image_root, thing_classes),
    )

    MetadataCatalog.get(name).set(
        json_file=json_path,
        image_root=image_root,
        evaluator_type="coco_taco",
        dirname="datasets/cigbutts",
        thing_classes=metadata["thing_classes"],
        thing_dataset_id_to_contiguous_id=metadata["thing_dataset_id_to_contiguous_id"],
    )   

