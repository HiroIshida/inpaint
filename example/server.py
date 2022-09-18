import argparse
from typing import Type

from inpaint.common import RequestData, ResponseData
from inpaint.server import InpaintPostHandler, TorchCallHandler, run_server


class HandlerMock(InpaintPostHandler):
    def handle_request(self, req: RequestData) -> ResponseData:
        resp = ResponseData(req.image, {})
        return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="test")
    args = parser.parse_args()
    is_testing = args.test

    handler: Type[InpaintPostHandler]
    if is_testing:
        handler = HandlerMock
    else:
        handler = TorchCallHandler
    run_server(handler)
