import json
from collections import defaultdict
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_JSON = ""
OUTPUT_JSON = ""

#iran or afghanistan

KEYWORD = ""
VALID_MONTH_PREFIXES = ()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def title_mentions_keyword(title: str, keyword: str) -> bool:
    if not title:
        return False
    return keyword.lower() in title.lower()


def is_in_target_period(published_at: str) -> bool:
    if not published_at:
        return False
    return published_at.startswith(VALID_MONTH_PREFIXES)


def get_channel_name(channel_entry: dict) -> str:
    return (
        channel_entry.get("channel_title")
        or channel_entry.get("input_name")
        or channel_entry.get("resolved_channel_id")
        or "UNKNOWN_CHANNEL"
    )


def main():
    input_path = Path(INPUT_JSON)
    if not input_path.exists():
        raise SystemExit(f"not found: {INPUT_JSON}")

    data = load_json(INPUT_JSON)
    channels = data.get("channels", [])

    filtered_channels = []
    counts_by_channel = defaultdict(int)
    total_videos = 0

    for channel in channels:
        channel_name = get_channel_name(channel)
        videos = channel.get("videos", [])

        matched_videos = []
        for video in videos:
            title = video.get("title", "")
            published_at = video.get("published_at", "")

            if title_mentions_keyword(title, KEYWORD) and is_in_target_period(published_at):
                matched_videos.append(video)

        if matched_videos:
            counts_by_channel[channel_name] = len(matched_videos)
            total_videos += len(matched_videos)

            filtered_channels.append(
                {
                    "channel_title": channel_name,
                    "input_name": channel.get("input_name"),
                    "input_url": channel.get("input_url"),
                    "resolved_channel_id": channel.get("resolved_channel_id"),
                    "videos_count": len(matched_videos),
                    "videos": matched_videos,
                }
            )

    output_data = {
        "source_file": INPUT_JSON,
        "filter": {
            "keyword_in_title": KEYWORD,
            "months": ["2021-08", "2021-09", "2021-10"],
        },
        "total_channels_with_matches": len(filtered_channels),
        "total_videos": total_videos,
        "channels": filtered_channels,
    }

    save_json(output_data, OUTPUT_JSON)



    for channel_name, count in sorted(counts_by_channel.items(), key=lambda x: (-x[1], x[0].lower())):
        print(f"{channel_name}: {count}")

    print()



if __name__ == "__main__":
    main()