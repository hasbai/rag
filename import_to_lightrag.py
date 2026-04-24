import json
from collections.abc import Iterable
from datetime import datetime, timedelta
from itertools import groupby

import numpy as np

from embd import embed

INPUT_FILE = "result.json"

MAX_MSGS = 100  # sessions larger than this get re-split
MIN_TEXT_LEN = 5  # skip individual messages shorter than this


def extract_text(raw) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    return ""


def load_messages(path: str) -> tuple[str, list[dict]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["name"], data["messages"]


def clean_and_format(messages: list[dict]) -> list[dict]:
    def _map(m: dict) -> dict:
        m["date"] = datetime.fromtimestamp(int(m["date_unixtime"]))
        m["sender"] = m.get("from", "unknown")
        m["text"] = extract_text(m.get("text"))
        return m

    messages = filter(lambda m: m.get("type") == "message" and m.get("text"), messages)
    messages = list(map(_map, messages))
    # Format
    result = []
    previous = None
    for m in messages:
        if m.get("forwarded_from"):
            m["sender"] += f" (Forwarded from {m['forwarded_from']})"
        if m.get("reply_to_message_id"):
            reply = next(
                filter(lambda msg: msg["id"] == m["reply_to_message_id"], messages),
                None,
            )
            if reply:
                m["text"] = f"(Reply to {reply['sender']}: {reply['text']}) {m['text']}"
        if (
            previous
            and previous["sender"] == m["sender"]
            and m["date"] - previous["date"] < timedelta(minutes=60)
        ):
            result[-1]["text"] += "\n" + m["text"]
        else:
            result.append(m)
        previous = m

    return result


def split_sessions(
    messages: list[dict],
    split_threshold: float = 0.6,
    window: int = 5,
    time_threshold: int = 12,
) -> list[list[dict]]:
    def _sliding_window(
        messages: list[dict], window: int = 100, overlap: int = 5
    ) -> list[list[dict]]:
        if len(messages) <= window:
            return [messages]

        chunks = []
        step = window - overlap
        for i in range(0, len(messages), step):
            chunk = messages[i : i + window]
            chunks.append(chunk)
            if i + window >= len(messages):
                break
        return chunks

    def _merge_small(lst: list[Iterable], min_seg_size: int = 2):
        result = []
        for is_single, group in groupby(lst, key=lambda x: len(x) <= min_seg_size):
            items = list(group)
            if is_single:
                result.append([elem for sublist in items for elem in sublist])
            else:
                result.extend(items)
        return result

    sessions, cur = [], [messages[0]]

    for i in range(1, len(messages)):
        time_gap = messages[i]["date"] - messages[i - 1]["date"]

        # 语义得分
        score = np.dot(embed(cur[-window:]), embed(messages[i : i + window]))

        if score < split_threshold or time_gap >= timedelta(hours=time_threshold):
            sessions.append(cur)
            cur = [messages[i]]
        else:
            cur.append(messages[i])

        print(f"\r{i}/{len(messages)} {score:.2f} {len(cur)}", end="", flush=True)

    sessions.append(cur)

    sessions = _merge_small(sessions)

    # Step 2: re-split oversized sessions in a 100 window
    final = []
    for session in sessions:
        if len(session) > MAX_MSGS:
            final.extend(_sliding_window(session))
        else:
            final.append(session)

    return final


def format_session(session: list[dict]) -> str:
    # lines = [f"[{session[0]['date'].isoformat()} - {session[-1]['date'].isoformat()}]"]
    lines = map(lambda m: f"{m['sender']}: {m['text']}", session)
    # lines.append("\n=====CHUNK=====\n")
    return "\n".join(lines)


def main(start=datetime.fromtimestamp(0), end=datetime.now()):
    print(f"Loading {INPUT_FILE} ...")
    chat_name, raw = load_messages(INPUT_FILE)
    cleaned = clean_and_format(raw)
    msgs = list(
        filter(
            lambda m: (
                m["date"] >= start - timedelta(days=1)
                and m["date"] <= end + timedelta(days=1)
            ),
            cleaned,
        )
    )
    print(f"Chat: {chat_name}")
    print(f"Processing {len(msgs)} of {len(cleaned)} messeges ...")

    # sessions = split_sessions(msgs)
    sessions = [msgs]
    sizes = list(map(lambda s: sum(map(lambda m: len(m["text"]), s)), sessions))

    print("\nSession split results:")
    print(f"  Sessions:       {len(sessions)}")
    print(f"  Min chars:      {min(sizes)}")
    print(f"  Median chars:   {sorted(sizes)[len(sizes) // 2]}")
    print(f"  Max chars:      {max(sizes)}")
    print(f"  Total chars:    {sum(sizes):,}")
    print()

    with open(
        f"{chat_name}[{start:%Y-%m-%d}~{end:%Y-%m-%d}].txt", "w", encoding="utf-8"
    ) as f:
        for session in sessions:
            f.write(format_session(session))
            f.write("\n=====CHUNK=====\n")
    return


if __name__ == "__main__":
    main(start=datetime(2024, 1, 1), end=datetime(2025, 1, 1))
