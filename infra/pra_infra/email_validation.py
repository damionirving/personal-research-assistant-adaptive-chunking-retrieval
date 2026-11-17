import re


def validate_owner_email(value: object) -> str:
    """Validate the explicitly configured SNS email subscription endpoint."""
    if not isinstance(value, str) or "\n" in value or "\r" in value:
        raise ValueError("Set notifications.owner_email or CDK context owner_email to a valid email address.")
    email = value.strip()
    pattern = r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
    if (not re.fullmatch(pattern, email) or
            re.search(r"cost_alert_email|replace_with|placeholder|@example\.(com|org|net)$", email, re.I)):
        raise ValueError("Set notifications.owner_email or CDK context owner_email to a valid non-placeholder email address.")
    return email
