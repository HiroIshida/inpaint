from inpaint.common import RequestData, ResponseData
from inpaint.server import InpaintServerBase, run_server


class InpaintServerMock(InpaintServerBase):
    def handle_request(self, req: RequestData) -> ResponseData:
        resp = ResponseData(req.image, {})
        return resp


if __name__ == "__main__":
    run_server(InpaintServerMock)
