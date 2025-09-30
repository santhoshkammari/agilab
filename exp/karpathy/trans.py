import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import json
import re
import time
import random

# Playlist link
playlist_url = "https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&si=evHU8u5feq9wKs4X"

# Collect all transcripts
all_transcripts = []

# Extract video IDs using yt-dlp
ydl_opts = {
    'quiet': True,
    'extract_flat': True,
    'flat_playlist': True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    playlist_info = ydl.extract_info(playlist_url, download=False)

for i, entry in enumerate(playlist_info['entries']):
    if entry:
        video_id = entry['id']
        title = entry.get('title', f'Video {video_id}')
        url = f"https://www.youtube.com/watch?v={video_id}"

        print(f"Processing {i+1}/{len(playlist_info['entries'])}: {title}")

        try:
            # Add delay to avoid rate limiting
            if i > 0:
                delay = random.uniform(3, 8)  # Random delay between 3-8 seconds
                print(f"  Waiting {delay:.1f}s to avoid rate limiting...")
                time.sleep(delay)

            # Try to get transcript with multiple approaches
            transcript = None

            # First try: simple get_transcript
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                print(f"  ✔ Got transcript using default method")
            except Exception as e:
                print(f"  ⚠ Default method failed: {str(e)[:100]}...")

            # Second try: list transcripts and pick the first available
            if transcript is None:
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    for transcript_obj in transcript_list:
                        try:
                            transcript = transcript_obj.fetch()
                            print(f"  ✔ Got transcript in {transcript_obj.language_code}")
                            break
                        except Exception as fetch_error:
                            print(f"  ⚠ Failed to fetch {transcript_obj.language_code}: {str(fetch_error)[:50]}...")
                            continue
                except Exception as list_error:
                    print(f"  ⚠ Failed to list transcripts: {str(list_error)[:100]}...")

            if transcript:
                all_transcripts.append({
                    "title": title,
                    "video_id": video_id,
                    "url": url,
                    "transcript": transcript
                })
                print(f"  ✔ Saved transcript with {len(transcript)} entries")
            else:
                print(f"  ✘ No transcript available")

        except Exception as e:
            print(f"  ✘ Error: {str(e)[:100]}...")

        print()  # Empty line for readability

# Write everything into ONE JSON file
with open("playlist_transcripts.json", "w", encoding="utf-8") as f:
    json.dump(all_transcripts, f, ensure_ascii=False, indent=2)

print("✅ All transcripts saved into playlist_transcripts.json")

