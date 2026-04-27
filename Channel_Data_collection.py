import json
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# =========================
# CONFIG (EDIT ME)
# =========================
API_KEY = ""
OUTPUT_JSON = ""
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.05


FETCH_VIDEO_DETAILS = True
VIDEO_DETAILS_BATCH_SIZE = 50


TARGET_MONTH_PREFIXES = ()

STOP_BEFORE_MONTH = ""

CHANNELS = [

]

BASE_URL = "https://www.googleapis.com/youtube/v3"


class YouTubeAPIError(Exception):
    pass


def api_get(endpoint: str, params: Dict) -> Dict:
    params = dict(params)
    params["key"] = API_KEY
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise YouTubeAPIError(f"{endpoint} failed ({resp.status_code}): {resp.text[:500]}")
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return resp.json()


def save_payload(payload: Dict, path: str = OUTPUT_JSON) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_initial_payload() -> Dict:
    return {
        "source": "YouTube Data API v3",
        "channels_requested": len(CHANNELS),
        "channels_succeeded": 0,
        "channels_failed": 0,
        "target_months": list(TARGET_MONTH_PREFIXES),
        "stop_before_month": STOP_BEFORE_MONTH,
        "failures": [],
        "channels": [],
        "last_updated_unix": time.time(),
    }


def update_and_save_payload(payload: Dict) -> None:
    payload["channels_succeeded"] = len(payload["channels"])
    payload["channels_failed"] = len(payload["failures"])
    payload["last_updated_unix"] = time.time()
    save_payload(payload)


def extract_year_month(published_at: Optional[str]) -> Optional[str]:
    if not published_at or len(published_at) < 7:
        return None
    return published_at[:7]


def should_stop_collecting(published_at: Optional[str]) -> bool:
    ym = extract_year_month(published_at)
    if ym is None:
        return False
    return ym < STOP_BEFORE_MONTH


def should_keep_video(published_at: Optional[str]) -> bool:
    ym = extract_year_month(published_at)
    if ym is None:
        return False
    return ym in TARGET_MONTH_PREFIXES


def normalize_youtube_path(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path.strip("/")


def parse_channel_hint(entry: Dict) -> Tuple[str, str]:
    path = normalize_youtube_path(entry["url"])
    if not path:
        raise ValueError(f"Cannot parse channel URL: {entry['url']}")

    parts = path.split("/")
    first = parts[0]

    if first == "channel" and len(parts) >= 2:
        return "channel_id", parts[1]
    if first.startswith("@"):
        return "handle", first[1:]
    if first == "user" and len(parts) >= 2:
        return "username", parts[1]
    if first == "c" and len(parts) >= 2:
        return "custom_url", parts[1]

    return "custom_url", first


def channels_list_by_id(channel_id: str) -> Optional[Dict]:
    data = api_get(
        "channels",
        {
            "part": "snippet,contentDetails,statistics,status,topicDetails",
            "id": channel_id,
            "maxResults": 1,
        },
    )
    items = data.get("items", [])
    return items[0] if items else None


def channels_list_by_handle(handle: str) -> Optional[Dict]:
    data = api_get(
        "channels",
        {
            "part": "snippet,contentDetails,statistics,status,topicDetails",
            "forHandle": handle,
            "maxResults": 1,
        },
    )
    items = data.get("items", [])
    return items[0] if items else None


def channels_list_by_username(username: str) -> Optional[Dict]:
    data = api_get(
        "channels",
        {
            "part": "snippet,contentDetails,statistics,status,topicDetails",
            "forUsername": username,
            "maxResults": 1,
        },
    )
    items = data.get("items", [])
    return items[0] if items else None


def score_candidate(candidate: Dict, target_name: str, target_token: str, target_url: str) -> int:
    score = 0
    snippet = candidate.get("snippet", {})
    title = (snippet.get("title") or "").lower()
    custom_url = (snippet.get("customUrl") or "").lower().lstrip("@")
    target_name_l = target_name.lower()
    target_token_l = target_token.lower().lstrip("@")
    target_url_l = target_url.lower()

    if title == target_name_l:
        score += 100
    if target_name_l in title:
        score += 50
    if custom_url == target_token_l:
        score += 120
    if target_token_l and target_token_l in custom_url:
        score += 40
    if custom_url and (f"/{custom_url}" in target_url_l or f"/@{custom_url}" in target_url_l):
        score += 60

    return score


def resolve_channel(entry: Dict) -> Dict:
    kind, value = parse_channel_hint(entry)

    if kind == "channel_id":
        item = channels_list_by_id(value)
        if item:
            return item

    if kind == "handle":
        item = channels_list_by_handle(value)
        if item:
            return item

    if kind == "username":
        item = channels_list_by_username(value)
        if item:
            return item

    if kind == "custom_url":
        item = channels_list_by_username(value)
        if item:
            return item

        data = api_get(
            "search",
            {
                "part": "snippet",
                "q": entry["name"],
                "type": "channel",
                "maxResults": 5,
            },
        )
        candidates = data.get("items", [])
        best_channel_id = None
        best_score = -1

        for c in candidates:
            score = score_candidate(c, entry["name"], value, entry["url"])
            if score > best_score:
                best_score = score
                best_channel_id = c.get("snippet", {}).get("channelId") or c.get("id", {}).get("channelId")

        if best_channel_id:
            item = channels_list_by_id(best_channel_id)
            if item:
                return item

    raise YouTubeAPIError(f"Could not resolve channel: {entry['name']} ({entry['url']})")


def get_uploads_playlist_id(channel_resource: Dict) -> str:
    return channel_resource["contentDetails"]["relatedPlaylists"]["uploads"]


def estimate_playlist_pages(playlist_id: str) -> Optional[int]:
    try:
        data = api_get(
            "playlistItems",
            {
                "part": "id",
                "playlistId": playlist_id,
                "maxResults": 1,
            },
        )
        page_info = data.get("pageInfo", {})
        total_results = page_info.get("totalResults")
        if isinstance(total_results, int) and total_results >= 0:
            return (total_results + 49) // 50
    except Exception:
        pass
    return None


def list_all_uploaded_videos(
    uploads_playlist_id: str,
    payload: Dict,
    channel_result_ref: Dict,
    channel_name_for_bar: str,
) -> List[Dict]:
    videos = []
    page_token = None
    stop_collection = False

    total_pages = estimate_playlist_pages(uploads_playlist_id)
    pbar = tqdm(
        total=total_pages,
        desc=f"Pages - {channel_name_for_bar[:35]}",
        unit="page",
        leave=False,
    )

    while True:
        params = {
            "part": "snippet,contentDetails,status",
            "playlistId": uploads_playlist_id,
            "maxResults": 50,
        }
        if page_token:
            params["pageToken"] = page_token

        data = api_get("playlistItems", params)
        items = data.get("items", [])

        for item in items:
            snippet = item.get("snippet", {})
            resource_id = snippet.get("resourceId", {})
            video_id = resource_id.get("videoId")
            if not video_id:
                continue

            published_at = snippet.get("publishedAt")

            if should_stop_collecting(published_at):
                stop_collection = True
                break

            if should_keep_video(published_at):
                videos.append(
                    {
                        "video_id": video_id,
                        "title": snippet.get("title"),
                        "published_at": published_at,
                        "description": snippet.get("description"),
                        "channel_id": snippet.get("channelId"),
                        "channel_title": snippet.get("channelTitle"),
                        "position_in_uploads_playlist": snippet.get("position"),
                        "playlist_item_status": item.get("status", {}),
                        "video_link": f"https://www.youtube.com/watch?v={video_id}",
                        "thumbnail": ((snippet.get("thumbnails") or {}).get("high") or {}).get("url"),
                    }
                )

        channel_result_ref["videos"] = videos
        channel_result_ref["videos_count_collected"] = len(videos)
        channel_result_ref["stopped_because_reached_older_than_stop_before_month"] = stop_collection
        update_and_save_payload(payload)

        pbar.update(1)

        if stop_collection:
            break

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    pbar.close()
    return videos


def chunked(seq: List[str], size: int) -> List[List[str]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def get_video_details_incremental(
    videos: List[Dict],
    payload: Dict,
    channel_result_ref: Dict,
    channel_name_for_bar: str,
) -> None:
    if not videos:
        return

    batches = chunked([v["video_id"] for v in videos], VIDEO_DETAILS_BATCH_SIZE)

    pbar = tqdm(
        total=len(batches),
        desc=f"Details - {channel_name_for_bar[:33]}",
        unit="batch",
        leave=False,
    )

    details_by_id: Dict[str, Dict] = {}

    for batch in batches:
        data = api_get(
            "videos",
            {
                "part": "snippet,contentDetails,statistics,status,topicDetails,liveStreamingDetails,recordingDetails",
                "id": ",".join(batch),
                "maxResults": len(batch),
            },
        )
        for item in data.get("items", []):
            details_by_id[item["id"]] = item

        for v in videos:
            if v["video_id"] in details_by_id:
                v["api_video_resource"] = details_by_id[v["video_id"]]

        channel_result_ref["videos"] = videos
        channel_result_ref["videos_count_collected"] = len(videos)
        update_and_save_payload(payload)

        pbar.update(1)

    pbar.close()


def collect_channel(entry: Dict, payload: Dict) -> Dict:
    channel = resolve_channel(entry)
    uploads_playlist_id = get_uploads_playlist_id(channel)
    snippet = channel.get("snippet", {})

    result = {
        "input_name": entry["name"],
        "input_url": entry["url"],
        "resolved_channel_id": channel.get("id"),
        "resolved_handle_or_custom_url": snippet.get("customUrl"),
        "channel_title": snippet.get("title"),
        "channel_description": snippet.get("description"),
        "channel_published_at": snippet.get("publishedAt"),
        "uploads_playlist_id": uploads_playlist_id,
        "channel_resource": channel,
        "videos_count_collected": 0,
        "stopped_because_reached_older_than_stop_before_month": False,
        "videos": [],
    }

    payload["channels"].append(result)
    update_and_save_payload(payload)

    videos = list_all_uploaded_videos(
        uploads_playlist_id=uploads_playlist_id,
        payload=payload,
        channel_result_ref=result,
        channel_name_for_bar=entry["name"],
    )

    if FETCH_VIDEO_DETAILS:
        get_video_details_incremental(
            videos=videos,
            payload=payload,
            channel_result_ref=result,
            channel_name_for_bar=entry["name"],
        )

    result["videos_count_collected"] = len(videos)
    result["videos"] = videos
    update_and_save_payload(payload)
    return result


def main() -> None:
    if API_KEY == "PUT_YOUR_YOUTUBE_API_KEY_HERE":
        raise SystemExit("Please edit API_KEY at the top of the script.")

    payload = build_initial_payload()
    update_and_save_payload(payload)

    channel_bar = tqdm(CHANNELS, desc="Channels", unit="channel")

    for entry in channel_bar:
        channel_bar.set_postfix_str(entry["name"][:40])

        try:
            collect_channel(entry, payload)
        except Exception as e:
            err = {
                "name": entry["name"],
                "url": entry["url"],
                "error": str(e),
            }
            payload["failures"].append(err)
            update_and_save_payload(payload)
            tqdm.write(f"FAILED: {entry['name']} -> {e}")

    channel_bar.close()

    update_and_save_payload(payload)

    print(f"\nSaved incrementally to {OUTPUT_JSON}")
    print(f"Channels succeeded: {payload['channels_succeeded']}")
    print(f"Channels failed: {payload['channels_failed']}")
    print(f"Collected only videos from: {', '.join(TARGET_MONTH_PREFIXES)}")


if __name__ == "__main__":
    main()