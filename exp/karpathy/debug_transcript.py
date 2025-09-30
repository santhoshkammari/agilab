from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

# Test with a single video
test_url = "https://www.youtube.com/watch?v=VMj-3S1tku0"  # First video from playlist

# Extract video ID
ydl_opts = {'quiet': True}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(test_url, download=False)
    video_id = info['id']
    title = info['title']

print(f"Video ID: {video_id}")
print(f"Title: {title}")

try:
    # Check what transcripts are available
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    print("\nAvailable transcripts:")
    for transcript in transcript_list:
        print(f"- Language: {transcript.language}")
        print(f"- Language code: {transcript.language_code}")
        print(f"- Is generated: {transcript.is_generated}")
        print(f"- Is translatable: {transcript.is_translatable}")
        print("---")

        # Try to fetch this transcript
        try:
            content = transcript.fetch()
            print(f"✔ Successfully fetched {transcript.language_code} transcript ({len(content)} entries)")
            if len(content) > 0:
                print(f"First entry: {content[0]}")
            break
        except Exception as e:
            print(f"✘ Failed to fetch {transcript.language_code}: {e}")

except Exception as e:
    print(f"Error listing transcripts: {e}")