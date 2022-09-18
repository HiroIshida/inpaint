from dataclasses import dataclass
from http.client import HTTPConnection
from typing import Optional

from inpaint.common import RequestData, ResponseData


@dataclass
class InpaintClient:
    host: str = "localhost"
    port: int = 8080
    conn: Optional[HTTPConnection] = None

    def __post_init__(self):
        if self.conn is None:
            self.conn = HTTPConnection(self.host, self.port)

    def __call__(self, req: RequestData) -> ResponseData:
        assert self.conn is not None
        headers = {"Content-type": "application/json"}
        self.conn.request("POST", "/post", req.serialize(), headers)
        response = self.conn.getresponse().read().decode()
        resp = ResponseData.deserialize(response)
        return resp
