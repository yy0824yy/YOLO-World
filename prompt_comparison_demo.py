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
IMAGE_PATH = "demo/sample_images/bus.jpg"
OUTPUT_DIR = "prompt_comparison_outputs"

PROMPT_GROUPS = {
    "prompt_a": ["person", "bus", "car"],
    "prompt_b": ["person", "bus", "wheel", "license plate", "glasses"],
    "prompt_c": ["vehicle", "human", "window", "wheel"],
}


def build_pipeline(cfg):
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
    return Compose(test_pipeline_cfg)


def infer(model, image_path, texts, test_pipeline, score_thr=0.20, max_dets=100):
    image = cv2.imread(image_path)
    image_rgb = image[:, :, [2, 1, 0]]
    data_info = dict(img=image_rgb, img_id=0, texts=texts)
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
    label_texts = [texts[int(label)][0] for label in labels]
    return image, boxes, labels, scores, label_texts


def draw_result(image, boxes, labels, scores, label_texts, title):
    canvas = image.copy()
    if len(boxes) > 0:
        detections = sv.Detections(xyxy=boxes, class_id=labels, confidence=scores)
        display_labels = [
            f"{label_text} {score:.2f}"
            for label_text, score in zip(label_texts, scores)
        ]
        canvas = sv.BoxAnnotator(thickness=2).annotate(
            scene=canvas, detections=detections
        )
        canvas = sv.LabelAnnotator(text_thickness=1, text_scale=0.45).annotate(
            scene=canvas, detections=detections, labels=display_labels
        )

    header_h = 54
    header = np.full((header_h, canvas.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (16, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([header, canvas])


def make_grid(images):
    target_h = 520
    resized = []
    for image in images:
        scale = target_h / image.shape[0]
        target_w = int(image.shape[1] * scale)
        resized.append(cv2.resize(image, (target_w, target_h)))

    gap = 16
    total_w = sum(img.shape[1] for img in resized) + gap * (len(resized) - 1)
    grid = np.full((target_h, total_w, 3), 245, dtype=np.uint8)
    x = 0
    for image in resized:
        grid[:, x : x + image.shape[1]] = image
        x += image.shape[1] + gap
    return grid


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg = Config.fromfile(CONFIG_FILE)
    cfg.work_dir = "./work_dirs"
    cfg.load_from = CHECKPOINT
    model = init_detector(cfg, checkpoint=CHECKPOINT, device="cuda:0", palette="coco")
    test_pipeline = build_pipeline(cfg)

    rendered = []
    for name, prompts in PROMPT_GROUPS.items():
        texts = [[prompt] for prompt in prompts] + [[" "]]
        image, boxes, labels, scores, label_texts = infer(
            model, IMAGE_PATH, texts, test_pipeline
        )
        title = f"{name.upper()}: {', '.join(prompts)}"
        result = draw_result(image, boxes, labels, scores, label_texts, title)
        output_path = osp.join(OUTPUT_DIR, f"bus_{name}.jpg")
        cv2.imwrite(output_path, result)
        rendered.append(result)

        print(f"\n{name}: {prompts}")
        print(f"output: {osp.abspath(output_path)}")
        if len(boxes) == 0:
            print("no detections")
        for idx, (box, label_text, score) in enumerate(zip(boxes, label_texts, scores), 1):
            print(
                f"{idx:02d}. {label_text:<14} score={float(score):.3f} box={np.round(box).astype(int).tolist()}"
            )

    grid = make_grid(rendered)
    cv2.imwrite(osp.join(OUTPUT_DIR, "bus_prompt_comparison.jpg"), grid)
    print(f"\ncomparison: {osp.abspath(osp.join(OUTPUT_DIR, 'bus_prompt_comparison.jpg'))}")


if __name__ == "__main__":
    main()
