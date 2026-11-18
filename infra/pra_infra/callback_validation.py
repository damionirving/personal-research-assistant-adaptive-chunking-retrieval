from urllib.parse import urlsplit


def validate_hosted_ui_callback_url(value: object) -> str:
    message = "Set CDK context -c hosted_ui_callback_url=<VALID_HTTPS_CALLBACK_URL>."
    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)
    value = value.strip()
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.fragment or any(c.isspace() for c in value) or parsed.port is not None and not 1 <= parsed.port <= 65535):
            raise ValueError(message)
    except ValueError as exc:
        raise ValueError(message) from exc
    return value
