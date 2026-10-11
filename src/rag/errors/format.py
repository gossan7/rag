from pydantic import ValidationError

MAX_DETAILS = 8
MAX_INPUT_LENGTH = 60


def format_validation_error(error: ValidationError) -> str:
    lines: list[str] = []
    for err in error.errors():
        loc = [str(part) for part in err["loc"]]
        msg = err["msg"].removeprefix("Value error, ")
        msg = msg.removeprefix("Assertion failed, ")

        if err["type"] == "extra_forbidden":
            msg = "unknown field"
        elif err["type"] == "missing":
            msg = "required field is missing"
        elif loc and loc[-1] == "[key]":
            loc.pop()
            msg = f"invalid name. {msg}"
        elif err["type"] not in ("value_error", "assertion_error"):
            got = repr(err["input"])
            if len(got) > MAX_INPUT_LENGTH:
                got = got[:MAX_INPUT_LENGTH] + "..."
            msg = f"{msg} (got {got})"

        if loc:
            details = [f"- '{'.'.join(loc)}': {msg}"]
        else:
            details = [f"- {line}" for line in msg.splitlines()]
        for line in details:
            if line not in lines:
                lines.append(line)

    if len(lines) > MAX_DETAILS:
        hidden = len(lines) - MAX_DETAILS
        lines = lines[:MAX_DETAILS] + [f"- ... and {hidden} more"]
    return "\n".join(lines)
