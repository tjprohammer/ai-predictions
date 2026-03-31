from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import date, datetime, timedelta
from typing import Iterator


def as_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def add_date_range_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--target-date", help="Single target date in YYYY-MM-DD format")
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD format")
    return parser


def resolve_date_range(args: Namespace, default_days_back: int = 0) -> tuple[date, date]:
    if getattr(args, "target_date", None):
        target = as_date(args.target_date)
        return target, target
    if getattr(args, "start_date", None) and getattr(args, "end_date", None):
        return as_date(args.start_date), as_date(args.end_date)
    end_date = date.today()
    start_date = end_date - timedelta(days=default_days_back)
    return start_date, end_date


def date_range(start_date: date, end_date: date) -> Iterator[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)