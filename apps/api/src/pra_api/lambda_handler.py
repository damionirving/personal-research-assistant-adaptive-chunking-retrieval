from mangum import Mangum

from pra_api.main import app
from pra_core.config import get_settings

settings = get_settings()

handler = Mangum(
    app,
    api_gateway_base_path=settings.api_root_path or "/",
)
