import argparse

from inpaint.common import RequestData, ResponseData
from inpaint.server import InpaintServerBase, TorchCallInpaintServer, run_server


class InpaintServerMock(InpaintServerBase):
    def handle_request(self, req: RequestData) -> ResponseData:
        resp = ResponseData(req.image, {})
        return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="test")
    args = parser.parse_args()
    is_testing = args.test

    if is_testing:
        handler = InpaintServerMock
    else:
        handler = TorchCallInpaintServer
    run_server(handler)
