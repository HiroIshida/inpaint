import numpy as np

from inpaint.client import InpaintClient
from inpaint.common import RequestData

if __name__ == "__main__":
    client = InpaintClient()

    image = np.random.randint(0, 255, (112, 122, 3), dtype=np.uint8)
    mask = np.zeros((112, 122, 3), dtype=np.uint8)
    mask[0:10, 0:10] = 1
    req = RequestData(image, mask, {})
    for _ in range(20):
        resp = client(req)
        print(resp)
