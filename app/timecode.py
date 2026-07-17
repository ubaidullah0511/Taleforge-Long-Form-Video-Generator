def parse_timestamp(ts: str) -> int:
    hours, minutes, seconds = (int(part) for part in ts.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def duration_seconds(start: str, end: str) -> float:
    return float(parse_timestamp(end) - parse_timestamp(start))
