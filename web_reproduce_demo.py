import os
import os.path as osp

import cv2
import gradio as gr
import numpy as np
import torch
import supervision as sv
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg


CONFIG_FILE = "configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
CHECKPOINT = "weights/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth"
DEFAULT_PROMPTS = "person, bus, wheel, license plate, glasses"
EXAMPLE_IMAGE = "demo/sample_images/bus.jpg"


MODEL = None
PIPELINE = None


def parse_prompts(prompt_text):
    prompts = []
    for raw_part in prompt_text.replace("\n", ",").split(","):
        part = raw_part.strip()
        if part:
            prompts.append(part)
    if not prompts:
        prompts = ["person", "bus"]
    return prompts


def load_model():
    global MODEL, PIPELINE
    if MODEL is not None and PIPELINE is not None:
        return MODEL, PIPELINE

    cfg = Config.fromfile(CONFIG_FILE)
    cfg.work_dir = "./work_dirs"
    cfg.load_from = CHECKPOINT
    MODEL = init_detector(cfg, checkpoint=CHECKPOINT, device="cuda:0", palette="coco")

    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
    PIPELINE = Compose(test_pipeline_cfg)
    return MODEL, PIPELINE


def run_detection(image, prompt_text, score_thr, max_dets):
    if image is None:
        return None, "Please upload an image first."

    model, test_pipeline = load_model()
    prompts = parse_prompts(prompt_text)
    texts = [[prompt] for prompt in prompts] + [[" "]]

    image_rgb = np.array(image)
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

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if len(boxes) > 0:
        detections = sv.Detections(xyxy=boxes, class_id=labels, confidence=scores)
        display_labels = [
            f"{label_text} {float(score):.2f}"
            for label_text, score in zip(label_texts, scores)
        ]
        image_bgr = sv.BoxAnnotator(thickness=2).annotate(
            scene=image_bgr.copy(), detections=detections
        )
        image_bgr = sv.LabelAnnotator(text_thickness=1, text_scale=0.5).annotate(
            scene=image_bgr, detections=detections, labels=display_labels
        )

    result_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if len(boxes) == 0:
        table = "No detections above threshold."
    else:
        rows = ["| # | label | score | box xyxy |", "| --- | --- | --- | --- |"]
        for idx, (box, label_text, score) in enumerate(zip(boxes, label_texts, scores), 1):
            rounded_box = np.round(box).astype(int).tolist()
            rows.append(f"| {idx} | {label_text} | {float(score):.3f} | {rounded_box} |")
        table = "\n".join(rows)

    return result_rgb, table


def build_demo():
    with gr.Blocks(title="YOLO-World Reproduction Demo") as demo:
        gr.Markdown("# YOLO-World Open-Vocabulary Detection Demo")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    label="Upload image",
                    type="pil",
                    value=EXAMPLE_IMAGE if osp.exists(EXAMPLE_IMAGE) else None,
                )
                prompt_text = gr.Textbox(
                    label="Prompts, separated by comma or newline",
                    lines=4,
                    value=DEFAULT_PROMPTS,
                )
                with gr.Row():
                    score_thr = gr.Slider(
                        0.0, 1.0, value=0.20, step=0.01, label="Score threshold"
                    )
                    max_dets = gr.Slider(
                        1, 100, value=50, step=1, label="Maximum detections"
                    )
                run_btn = gr.Button("Run detection", variant="primary")
            with gr.Column(scale=1):
                output_image = gr.Image(label="Detection result", type="numpy")
                output_table = gr.Markdown(label="Detection table")

        gr.Examples(
            examples=[
                [EXAMPLE_IMAGE, "person, bus, car", 0.20, 50],
                [EXAMPLE_IMAGE, "person, bus, wheel, license plate, glasses", 0.20, 50],
                [EXAMPLE_IMAGE, "vehicle, human, window, wheel", 0.20, 50],
                ["third_party/mmyolo/demo/dog.jpg", "dog, bicycle, wheel, car", 0.20, 50],
            ],
            inputs=[image, prompt_text, score_thr, max_dets],
        )

        run_btn.click(
            run_detection,
            inputs=[image, prompt_text, score_thr, max_dets],
            outputs=[output_image, output_table],
        )
    return demo


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    app = build_demo()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
