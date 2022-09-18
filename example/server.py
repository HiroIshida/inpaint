import argparse
from typing import Type

from inpaint.common import RequestData, ResponseData
from inpaint.server import InpaintPostHandler, TorchCallHandler, run_server


class HandlerMock(InpaintPostHandler):
    def handle_request(self, req: RequestData) -> ResponseData:
        resp = ResponseData(req.image, req.image)
        return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="test")
    parser.add_argument("-port", type=int, default=8080, help="port")
    args = parser.parse_args()
    is_testing: bool = args.test
    port: int = args.port

    handler: Type[InpaintPostHandler]
    if is_testing:
        handler = HandlerMock
    else:
        handler = TorchCallHandler
    run_server(handler, port=port)
