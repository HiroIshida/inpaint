import logging
from abc import abstractmethod
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Type

from inpaint.common import RequestData, ResponseData


class InpaintServerBase(BaseHTTPRequestHandler):
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


class TorchCallInpaintServer(InpaintServerBase):
    def handle_request(self, req: RequestData) -> ResponseData:

        resp = ResponseData(req.image, {})
        return resp


def run_server(handler_class: Type[InpaintServerBase], server_class=HTTPServer, port=8080):
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
