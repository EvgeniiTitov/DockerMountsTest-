import os
import typing

import cv2

from model import ComponentsModel
from utils import draw_boxes


ALLOWED_EXTS = [".png", ".jpg", ".jpeg"]


def get_items_to_process() -> typing.Generator:
    if not len(os.listdir("source")):
        raise FileNotFoundError("No images to process in the source folder")
    for filename in os.listdir("source"):
        if any(filename.endswith(ext.lower()) for ext in ALLOWED_EXTS):
            yield os.path.join("source", filename)


def main():
    try:
        model = ComponentsModel()
    except Exception as e:
        print(f"Failed to init the model. Error: {e}")
        raise e
    for item in get_items_to_process():
        image = cv2.imread(item)
        if image is None:
            print(f"Failed to read image {image}")
            continue
        preds = model.predict(image)
        if len(preds):
            draw_boxes(image, preds)
        image_name = os.path.basename(item)
        cv2.imwrite(os.path.join("result", image_name), image)
        print("Processed image:", image_name)


if __name__ == "__main__":
    main()
