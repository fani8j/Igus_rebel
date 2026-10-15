"""Per-run access token and request origin checks for the calibration UI (no ROS imports)."""
import hmac
import re
import secrets
from typing import Optional

PAGE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


class Auth:
    """A random token generated at start (``token`` may be injected for tests).  The page sends it in ``X-Token``."""

    def __init__(self, port: int, token: Optional[str] = None):
        self.port = int(port)
        self.token = token or secrets.token_urlsafe(24)

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?token={self.token}"

    def valid(self, header_value: Optional[str]) -> bool:
        if not header_value:
            return False
        return hmac.compare_digest(self.token.encode("utf-8"), header_value.encode("utf-8", "ignore"))

    def allowed_origins(self):
        return {f"http://127.0.0.1:{self.port}", f"http://localhost:{self.port}"}

    def origin_ok(self, origin: Optional[str]) -> bool:
        """A POST must carry no Origin header (curl, same-origin fetch in some browsers) or one of ours."""
        return origin is None or origin in self.allowed_origins()

    def host_ok(self, host: Optional[str]) -> bool:
        """Reject foreign Host headers (DNS rebinding); a missing Host (HTTP/1.0) is accepted."""
        return host is None or host in {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}

    @staticmethod
    def page_id_ok(page_id: Optional[str]) -> bool:
        return bool(page_id) and PAGE_ID_RE.match(page_id) is not None
