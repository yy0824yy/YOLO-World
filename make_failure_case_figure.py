import os
import os.path as osp

import cv2
import numpy as np


INPUT_IMAGE = "reproduce_outputs/demo.jpg"
OUTPUT_DIR = "failure_cases"
OUTPUT_IMAGE = "failure_case_dense_small_cars.jpg"


def put_lines(canvas, lines, x, y, scale=0.65, color=(40, 40, 40), thickness=2):
    line_h = int(30 * scale / 0.65)
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def wrap_line(line, max_chars):
    words = line.split()
    wrapped = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                wrapped.append(current)
            current = word
    if current:
        wrapped.append(current)
    return wrapped


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image = cv2.imread(INPUT_IMAGE)
    if image is None:
        raise FileNotFoundError(INPUT_IMAGE)

    target_w = 1040
    scale = target_w / image.shape[1]
    target_h = int(image.shape[0] * scale)
    image = cv2.resize(image, (target_w, target_h))

    top_h = 92
    bottom_h = 190
    margin = 22
    canvas = np.full(
        (image.shape[0] + top_h + bottom_h, image.shape[1], 3),
        248,
        dtype=np.uint8,
    )
    canvas[top_h : top_h + image.shape[0], :, :] = image

    put_lines(
        canvas,
        ["Failure / Limitation Case: distant dense vehicles"],
        margin,
        38,
        scale=0.82,
        color=(20, 20, 20),
        thickness=2,
    )

    notes = [
        "Observed issue: cars are small and densely arranged, so boxes and labels overlap.",
        "Confidence is relatively low: the highest car score is about 0.33.",
        "Reason: small targets and dense scenes are harder for open-vocabulary matching.",
        "Report usage: use this figure to discuss limitations, not as the main success case.",
    ]
    wrapped_notes = []
    for note in notes:
        wrapped_notes.extend(wrap_line(note, max_chars=108))
    put_lines(
        canvas,
        wrapped_notes,
        margin,
        top_h + image.shape[0] + 34,
        scale=0.58,
        color=(45, 45, 45),
        thickness=1,
    )

    cv2.rectangle(
        canvas,
        (0, top_h),
        (image.shape[1] - 1, top_h + image.shape[0] - 1),
        (80, 80, 80),
        2,
    )

    output_path = osp.join(OUTPUT_DIR, OUTPUT_IMAGE)
    cv2.imwrite(output_path, canvas)
    print(osp.abspath(output_path))


if __name__ == "__main__":
    main()
