from youtube_transcript_api import YouTubeTranscriptApi
import sys

def get_transcript(video_id, languages=None):
    """
    Fetches the transcript for a YouTube video by ID.
    :param video_id: str, YouTube video ID
    :param languages: list of str, language codes to try (e.g. ['en', 'en-US'])
    :return: list of dictionaries with 'text', 'start', 'duration'
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        return transcript
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    # For example video https://youtu.be/VMj-3S1tku0 the ID is VMj-3S1tku0
    video_id = "VMj-3S1tku0"
    # You can specify languages if you want, else it auto-detects
    languages = ['en']  # try English first

    transcript = get_transcript(video_id, languages)
    if transcript:
        for entry in transcript:
            start = entry['start']
            duration = entry['duration']
            text = entry['text']
            # Print in format: [start_time] text
            print(f"[{start:.2f}] {text}")
    else:
        print("Transcript not available.")

if __name__ == "__main__":
    main()

