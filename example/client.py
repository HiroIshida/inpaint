import numpy as np
from PIL import Image

from inpaint.client import InpaintClient
from inpaint.common import RequestData

if __name__ == "__main__":
    client = InpaintClient()

    image = np.array(Image.open("./example.png"))
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[100:200, 100:200] = 1
    req = RequestData(image, mask, {})
    resp = client(req)
    print(resp)
