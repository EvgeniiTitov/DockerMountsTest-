import typing

import cv2
import numpy as np


def draw_boxes(image: np.ndarray, preds: typing.List[list]) -> None:
    for pred in preds:
        cls, acc, left, top, right, bot = pred
        cv2.rectangle(
            image,
            (left, top),
            (right, bot),
            (0, 255, 0),
            2,
            cv2.FONT_HERSHEY_SIMPLEX,
        )
    return
