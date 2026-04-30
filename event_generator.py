"""Simple standalone event generator for the Gallery message pipeline.

Run this in a separate terminal while the main app is running:
    python main.py
    python event_generator.py --events upload,search,list --count 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from itertools import cycle

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import ImageListRequested, ImageUploadRequested, SearchRequested

_EVENT_NAMES = ("upload", "search", "list")
_DEFAULT_UPLOAD_PATHS = (
    "samples/bananas.jpg",
    "samples/apple.jpeg",
    "samples/tomato.png",
)
_OBSERVED_CHANNELS = (
    Channels.IMAGE_UPLOAD_REQUESTED,
    Channels.IMAGE_ACCEPTED,
    Channels.IMAGE_EMBEDDING_REQUESTED,
    Channels.IMAGE_STORED,
    Channels.IMAGE_PIPELINE_COMPLETE,
    Channels.IMAGE_LIST_REQUESTED,
    Channels.IMAGE_LIST_READY,
    Channels.SEARCH_REQUESTED,
    Channels.SEARCH_RESULTS_READY,
)


def parse_events(value: str) -> list[str]:
    events = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not events:
        raise ValueError("At least one event is required")
    invalid = [name for name in events if name not in _EVENT_NAMES]
    if invalid:
        allowed = ", ".join(_EVENT_NAMES)
        raise ValueError(f"Unsupported event(s): {', '.join(invalid)}. Allowed: {allowed}")
    return events


def build_trigger(
    event_name: str,
    *,
    upload_path: str,
    search_query: str,
    search_top_k: int,
) -> tuple[str, object]:
    if event_name == "upload":
        return Channels.IMAGE_UPLOAD_REQUESTED, ImageUploadRequested(path=upload_path)
    if event_name == "search":
        return (
            Channels.SEARCH_REQUESTED,
            SearchRequested(query=search_query, top_k=search_top_k),
        )
    if event_name == "list":
        return Channels.IMAGE_LIST_REQUESTED, ImageListRequested()
    raise ValueError(f"Unsupported event: {event_name}")


def _print_json(label: str, channel: str, payload: dict) -> None:
    print(f"{label} channel={channel} payload={json.dumps(payload, ensure_ascii=False)}")


async def run_generator(args: argparse.Namespace) -> None:
    broker = RedisBroker()
    await broker.connect()

    for observed_channel in _OBSERVED_CHANNELS:
        @broker.on(observed_channel)
        async def _on_message(msg: dict, *, ch: str = observed_channel) -> None:
            _print_json("[OBSERVED]", ch, msg)

    listener = asyncio.create_task(broker.listen())
    uploads = cycle(args.upload_paths)
    events_to_send = args.events

    try:
        for _ in range(args.count):
            for event_name in events_to_send:
                upload_path = next(uploads)
                channel, message = build_trigger(
                    event_name,
                    upload_path=upload_path,
                    search_query=args.query,
                    search_top_k=args.top_k,
                )
                payload = asdict(message)
                _print_json("[TRIGGER]", channel, payload)
                await broker.publish(channel, message)
                await asyncio.sleep(args.delay)

        await asyncio.sleep(args.observe_seconds)
    finally:
        listener.cancel()
        try:
            await listener
        except asyncio.CancelledError:
            pass
        await broker.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate simple Gallery broker events.")
    parser.add_argument(
        "--events",
        default="upload,search,list",
        help="Comma-separated events to trigger: upload,search,list",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="How many times to run the selected event sequence.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay in seconds between each triggered event.",
    )
    parser.add_argument(
        "--query",
        default="fruit",
        help="Search query used when triggering search events.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="top_k used when triggering search events.",
    )
    parser.add_argument(
        "--upload-paths",
        nargs="+",
        default=list(_DEFAULT_UPLOAD_PATHS),
        help="Space-separated upload paths, cycled for upload events.",
    )
    parser.add_argument(
        "--observe-seconds",
        type=float,
        default=1.0,
        help="How long to keep listening after triggers complete.",
    )
    return parser


async def _async_main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.events = parse_events(args.events)
    except ValueError as exc:
        parser.error(str(exc))
    if args.count < 1:
        parser.error("--count must be >= 1")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.delay < 0:
        parser.error("--delay must be >= 0")
    if args.observe_seconds < 0:
        parser.error("--observe-seconds must be >= 0")
    await run_generator(args)


if __name__ == "__main__":
    asyncio.run(_async_main())
