import logging
import subprocess
import tempfile
from abc import abstractmethod
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Type

import numpy as np
from PIL import Image

from inpaint.common import RequestData, ResponseData


class InpaintPostHandler(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length).decode("utf-8")
        req = RequestData.deserialize(post_data)
        self._set_response()
        resp = self.handle_request(req)
        self.wfile.write(resp.serialize().encode("utf-8"))

    @abstractmethod
    def handle_request(self, req: RequestData) -> ResponseData:
        pass


class TorchCallHandler(InpaintPostHandler):
    def handle_request(self, req: RequestData) -> ResponseData:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            image_path = (p / "image.png").expanduser()
            im = Image.fromarray(req.image)
            im.save(image_path)

            mask_path = (p / "mask.png").expanduser()
            mask = Image.fromarray(req.mask)
            mask.save(mask_path)

            here_path = Path(__file__).parent
            script_dir_path = (here_path / "siggraph2017_inpainting").absolute()
            script_path = script_dir_path / "inpaint.lua"
            cmd = "cd {siggraph} && th {script} --input {image} --mask {mask}".format(
                siggraph=script_dir_path, script=script_path, image=image_path, mask=mask_path
            )
            subprocess.run(cmd, shell=True)
            output_path = script_dir_path / "out.png"
            image_out = np.array(Image.open(output_path))
            output_path.unlink()

        resp = ResponseData(image_out)
        return resp


def run_server(handler_class: Type[InpaintPostHandler], server_class=HTTPServer, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    logging.info("Starting httpd...\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info("Stopping httpd...\n")


if __name__ == "__main__":
    run_server(TorchCallHandler)
