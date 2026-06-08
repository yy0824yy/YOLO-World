# Copyright (c) Tencent Inc. and Custom Reproduce Script. All rights reserved.
import os
import os.path as osp
import cv2
import torch
import numpy as np
import supervision as sv
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

def inference(model, image_path, texts, test_pipeline, score_thr=0.3, max_dets=100):
    image = cv2.imread(image_path)
    # Convert BGR to RGB
    image_rgb = image[:, :, [2, 1, 0]]
    data_info = dict(img=image_rgb, img_id=0, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    
    with torch.no_grad():
        output = model.test_step(data_batch)[0]
    
    pred_instances = output.pred_instances
    # Filter by score
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    
    # Cap maximum detections
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]
    
    pred_instances = pred_instances.cpu().numpy()
    boxes = pred_instances['bboxes']
    labels = pred_instances['labels']
    scores = pred_instances['scores']
    label_texts = [texts[x][0] for x in labels]
    return boxes, labels, label_texts, scores

if __name__ == "__main__":
    # 1. 设置使用的 YOLO-World-v2-S 配置文件与刚下载好的权重
    config_file = "configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    checkpoint = "weights/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth"
    
    if not osp.exists(checkpoint):
        raise FileNotFoundError(f"未找到权重文件: {checkpoint}，请先执行 wget 下载它！")

    print("正在加载 YOLO-World-v2-S 模型及权重...")
    cfg = Config.fromfile(config_file)
    cfg.work_dir = osp.join('./work_dirs')
    cfg.load_from = checkpoint
    
    # 初始化检测器并在 GPU 0 上运行 (我们的服务器有8张 4090！)
    model = init_detector(cfg, checkpoint=checkpoint, device='cuda:0', palette='coco')
    
    # 准备测试数据流 pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)
    
    # 2. 定义我们的 开放词汇 (Open-Vocabulary) 文本提示词
    # 我们可以自由定义任何想检测的东西，哪怕模型在 COCO 数据集里没见过！
    # 格式为 List[List[str]]，并在末尾留一个空类别作为背景/空检测占位
    texts = [
        ['person'], 
        ['bus'], 
        ['backpack'], 
        ['glasses'], 
        ['wheel'],
        ['license plate'],
        [' ']
    ]
    
    image_path = "demo/sample_images/bus.jpg"
    print(f"开始对图像进行开放词汇检测: {image_path}")
    print(f"目标检测类别: {[t[0] for t in texts[:-1]]}")
    
    boxes, labels, label_texts, scores = inference(model, image_path, texts, test_pipeline, score_thr=0.25)
    
    print("\n--- 检测结果 ---")
    for idx, (box, lbl_text, score) in enumerate(zip(boxes, label_texts, scores)):
        print(f"目标 #{idx+1}: 类别={lbl_text}, 置信度={score:.3f}, 边界框={np.round(box).tolist()}")
        
    # 3. 使用 supervision 库进行可视化画图
    image_bgr = cv2.imread(image_path)
    
    # 转换 boxes 为 supervision 格式
    if len(boxes) > 0:
        detections = sv.Detections(
            xyxy=boxes,
            class_id=labels,
            confidence=scores
        )
        
        # 定义画框器和贴标器
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.4)
        
        # 准备标签文本
        display_labels = [f"{lbl_text} {score:.2f}" for lbl_text, score in zip(label_texts, scores)]
        
        # 绘制
        annotated_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=display_labels)
    else:
        annotated_image = image_bgr
        print("未检测到任何目标！")

    # 保存检测结果图
    output_path = "../reproduce_result.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"\n🎉 完美复现！带有检测框的可视化图片已保存至: {osp.abspath(output_path)}")
