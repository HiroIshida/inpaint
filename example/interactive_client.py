import argparse
import copy
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from inpaint.client import InpaintClient
from inpaint.common import RequestData


class InteractiveTuner:
    client: InpaintClient
    img_original: np.ndarray
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    def __init__(self, client: InpaintClient, img_original: np.ndarray):

        shape = img_original.shape
        cv2.namedWindow("window")  # type: ignore
        cv2.namedWindow("window-inpainted")  # type: ignore

        cv2.createTrackbar("x_min", "window", int(shape[1] / 3), shape[1] - 1, lambda x: None)  # type: ignore
        cv2.createTrackbar("x_max", "window", int(2 * shape[1] / 3), shape[1], lambda x: None)  # type: ignore
        cv2.createTrackbar("y_min", "window", int(shape[0] / 3), shape[0] - 1, lambda x: None)  # type: ignore
        cv2.createTrackbar("y_max", "window", int(2 * shape[0] / 3), shape[0], lambda x: None)  # type: ignore

        self.client = client
        self.img_original = img_original

    def draw(self, use_gpu: bool, pretrained_model: Optional[str] = None) -> None:
        img_masked = self.apply(self.img_original)
        cv2.imshow("window", img_masked[..., ::-1])  # type: ignore

        mask = self.get_mask_image()

        metadata = {}
        if pretrained_model is not None:
            metadata["model"] = pretrained_model
        req = RequestData(self.img_original, mask, use_gpu=use_gpu, metadata=metadata)
        resp = self.client(req)
        cv2.imshow("window-inpainted", resp.image[..., ::-1])  # type: ignore

    def get_mask_image(self) -> np.ndarray:
        mask = np.zeros(self.img_original.shape[:2], dtype=np.uint8)
        mask[self.y_min : self.y_max, self.x_min : self.x_max] = 255
        return mask

    def reflect_trackbar(self):
        self.x_min = cv2.getTrackbarPos("x_min", "window")
        self.x_max = cv2.getTrackbarPos("x_max", "window")
        if self.x_max <= self.x_min:
            self.x_max = self.x_min + 1

        self.y_min = cv2.getTrackbarPos("y_min", "window")
        self.y_max = cv2.getTrackbarPos("y_max", "window")
        if self.y_max <= self.y_min:
            self.y_max = self.y_min + 1

    def apply(self, img: np.ndarray) -> np.ndarray:
        img_masked = copy.deepcopy(img)
        img_masked[self.y_min : self.y_max, self.x_min : self.x_max] = 255
        return img_masked


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="use gpu")
    parser.add_argument("-port", type=int, default=8080, help="port")
    parser.add_argument("-host", type=str, default="localhost", help="hostname")
    parser.add_argument("-image", type=str, help="image path")
    parser.add_argument("-model", type=str, help="pretrained model")

    args = parser.parse_args()
    use_gpu: bool = args.gpu
    port: int = args.port
    host: str = args.host
    image_path_str: Optional[str] = args.image
    pretrained_model: Optional[str] = args.model

    client = InpaintClient(host=host, port=port)

    base_path = Path(__file__).parent.absolute()
    if image_path_str is None:
        image_path = base_path / "example.png"
    else:
        image_path = Path(image_path_str).expanduser().absolute()
    image = np.array(Image.open(image_path))

    tuner = InteractiveTuner(client, image)
    while True:
        cv2.waitKey(10)
        tuner.reflect_trackbar()
        tuner.draw(use_gpu=use_gpu, pretrained_model=pretrained_model)
