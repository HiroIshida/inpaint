import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from inpaint.client import InpaintClient
from inpaint.common import RequestData

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="use gpu")
    parser.add_argument("-port", type=int, default=8080, help="port")
    parser.add_argument("-host", type=str, default="localhost", help="hostname")

    args = parser.parse_args()
    use_gpu: bool = args.gpu
    port: int = args.port
    host: str = args.host

    client = InpaintClient(host=host, port=port)

    image = np.array(Image.open("./example.png"))
    mask = np.array(Image.open("./example_mask.png"))
    req = RequestData(image, mask, {})
    req.use_gpu = use_gpu

    ts = time.time()
    resp = client(req)
    print("elapsed time: {} sec".format(time.time() - ts))

    fig, axes = plt.subplots(1, 3)

    axes[0].imshow(image)
    axes[0].set_title("original")
    axes[1].imshow(resp.debug_image)
    axes[1].set_title("mask")
    axes[2].imshow(resp.image)
    axes[2].set_title("inpainted image")

    plt.show()
