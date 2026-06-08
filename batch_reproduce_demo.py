import os
import os.path as osp

import cv2
import numpy as np
import torch
import supervision as sv
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg


CONFIG_FILE = "configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
CHECKPOINT = "weights/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth"
OUTPUT_DIR = "reproduce_outputs"

TEXTS = [
    ["person"],
    ["bus"],
    ["dog"],
    ["car"],
    ["bicycle"],
    ["horse"],
    ["backpack"],
    ["ball"],
    ["wheel"],
    ["license plate"],
    ["glasses"],
    [" "],
]

IMAGES = [
    "demo/sample_images/bus.jpg",
    "demo/sample_images/zidane.jpg",
    "third_party/mmyolo/demo/dog.jpg",
    "third_party/mmyolo/demo/demo.jpg",
    "third_party/mmyolo/demo/large_image.jpg",
]


def infer(model, image_path, test_pipeline, score_thr=0.20, max_dets=100):
    image = cv2.imread(image_path)
    image_rgb = image[:, :, [2, 1, 0]]
    data_info = dict(img=image_rgb, img_id=0, texts=TEXTS)
    data_info = test_pipeline(data_info)
    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    with torch.no_grad():
        output = model.test_step(data_batch)[0]

    pred_instances = output.pred_instances
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    boxes = pred_instances["bboxes"]
    labels = pred_instances["labels"]
    scores = pred_instances["scores"]
    label_texts = [TEXTS[int(label)][0] for label in labels]
    return image, boxes, labels, scores, label_texts


def draw(image, boxes, labels, scores, label_texts):
    if len(boxes) == 0:
        return image

    detections = sv.Detections(xyxy=boxes, class_id=labels, confidence=scores)
    display_labels = [
        f"{label_text} {score:.2f}" for label_text, score in zip(label_texts, scores)
    ]
    image = sv.BoxAnnotator(thickness=2).annotate(
        scene=image.copy(), detections=detections
    )
    image = sv.LabelAnnotator(text_thickness=1, text_scale=0.45).annotate(
        scene=image, detections=detections, labels=display_labels
    )
    return image


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = Config.fromfile(CONFIG_FILE)
    cfg.work_dir = "./work_dirs"
    cfg.load_from = CHECKPOINT
    model = init_detector(cfg, checkpoint=CHECKPOINT, device="cuda:0", palette="coco")

    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
    test_pipeline = Compose(test_pipeline_cfg)

    print("prompts:", [text[0] for text in TEXTS[:-1]])
    for image_path in IMAGES:
        image, boxes, labels, scores, label_texts = infer(model, image_path, test_pipeline)
        out_image = draw(image, boxes, labels, scores, label_texts)
        output_path = osp.join(OUTPUT_DIR, osp.basename(image_path))
        cv2.imwrite(output_path, out_image)

        print(f"\nimage: {image_path}")
        print(f"output: {osp.abspath(output_path)}")
        if len(boxes) == 0:
            print("no detections")
        for idx, (box, label_text, score) in enumerate(zip(boxes, label_texts, scores), 1):
            print(
                f"{idx:02d}. {label_text:<14} score={float(score):.3f} box={np.round(box).astype(int).tolist()}"
            )


if __name__ == "__main__":
    main()
