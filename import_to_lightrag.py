import json
from datetime import datetime, timedelta

import mlx.core as mx
import numpy as np
from mlx_embeddings import load

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


def load_messages(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    chat = data["chats"]["list"][0]
    return chat["name"], chat["messages"]


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
        m["sender"] += (
            f" (Forwarded from {m['forwarded_from']})"
            if m.get("forwarded_from")
            else ""
        )
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
            and m["date"] - previous["date"] < timedelta(minutes=5)
        ):
            result[-1]["text"] += "\n" + m["text"]
        else:
            result.append(m)
        previous = m

    return result


def _embed(messages: list[dict]) -> np.ndarray:
    text = "\n".join(map(lambda m: m["text"], messages))
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])
    output = model(input_ids)
    mx.eval(output.text_embeds)
    result = np.array(output.text_embeds[0])
    return result


def split_sessions(
    messages: list[dict],
    split_threshold: float = 0.6,
    window: int = 4,
    min_seg_size: int = 2,
) -> list[list[dict]]:
    def _split(messages: list[dict], threshold: timedelta) -> list[list[dict]]:
        sessions = []
        cur = [messages[0]]
        for m in messages[1:]:
            if m["date"] - cur[-1]["date"] >= threshold:
                sessions.append(cur)
                cur = [m]
            else:
                cur.append(m)
        sessions.append(cur)
        return sessions

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

    if len(messages) <= min_seg_size:
        return [messages]

    sessions, cur = [], [messages[0]]

    for i in range(1, len(messages)):
        time_gap = messages[i]["date"] - messages[i - 1]["date"]

        # 硬截断：超过 12h 一定切
        if time_gap >= timedelta(hours=12):
            sessions.append(cur)
            cur = [messages[i]]
            continue

        # 语义得分
        score = np.dot(_embed(cur[-window:]), _embed(messages[i : i + window]))

        if score < split_threshold and len(cur) >= min_seg_size:
            sessions.append(cur)
            cur = [messages[i]]
        else:
            cur.append(messages[i])

        print(f"\r{i}/{len(messages)} {score:.2f} {len(cur)}", end="", flush=True)

    sessions.append(cur)
    return sessions

    # Step 2: re-split oversized sessions in a 100 window
    final = []
    for s in messages:
        if len(s) <= MAX_MSGS:
            final.append(s)
        else:
            final.extend(_sliding_window(s))

    return final


def format_session(session: list[dict]) -> str:
    lines = [f"[{session[0]['date'].isoformat()} - {session[-1]['date'].isoformat()}]"]
    lines.extend(map(lambda m: f"{m['sender']}: {m['text']}", session))
    lines.append("\n=====CHUNK=====\n")
    return "\n".join(lines)


def main():
    print(f"Loading {INPUT_FILE} ...")
    chat_name, msgs = load_messages(INPUT_FILE)
    msgs = clean_and_format(msgs)
    print(f"  Chat: {chat_name}")
    print(f"  Messages: {len(msgs)}")

    sessions = split_sessions(msgs)
    sizes = [len(s) for s in sessions]
    total_chars = sum(
        sum(len(extract_text(m.get("text", "")).strip()) for m in s) for s in sessions
    )
    print("\nSession split results:")
    print(f"  Sessions:       {len(sessions)}")
    print(f"  Min size:       {min(sizes)} msgs")
    print(f"  Median size:    {sorted(sizes)[len(sizes) // 2]} msgs")
    print(f"  Max size:       {max(sizes)} msgs")
    print(f"  Total chars:    {total_chars:,}")
    print()

    with open(f"{chat_name}.txt", "w", encoding="utf-8") as f:
        for session in sessions:
            f.write(format_session(session))
    return


if __name__ == "__main__":
    model, tokenizer = load("mlx-community/bge-m3-mlx-4bit")
    main()
