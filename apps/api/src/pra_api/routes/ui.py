from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "templates" / "index.html"


@router.get("/", response_class=HTMLResponse)
def index() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")
