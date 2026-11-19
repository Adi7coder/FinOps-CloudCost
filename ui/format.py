from datetime import datetime

TS_FORMAT = "%Y-%m-%d %H:%M"


def inr(value, decimals: int = 0) -> str:
    if value is None:
        return "-"
    negative = value < 0
    whole, _, fraction = f"{abs(value):.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{'-' if negative else ''}₹{whole}{'.' + fraction if fraction else ''}"


def inr_compact(value) -> str:
    if value is None:
        return "-"
    size = abs(value)
    if size >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    if size >= 1e5:
        return f"₹{value / 1e5:.2f} L"
    if size >= 1e3:
        return f"₹{value / 1e3:.1f}K"
    return inr(value)


def pct(value, decimals: int = 1) -> str:
    return "-" if value is None else f"{value:.{decimals}f}%"


def ist(text: str, with_date: bool = True) -> str:
    if not text:
        return "-"
    moment = datetime.strptime(text, TS_FORMAT)
    return moment.strftime("%d %b %Y, %H:%M IST" if with_date else "%H:%M IST")


def day_label(text: str) -> str:
    return datetime.strptime(text, "%Y-%m-%d").strftime("%d %b %Y")


def duration(hours) -> str:
    if hours is None or hours <= 0:
        return "-"
    if hours >= 48:
        return f"{hours / 24:.1f} days"
    return f"{hours:.0f} h"


def title_case(text: str) -> str:
    return text.replace("_", " ").capitalize() if text else "-"
