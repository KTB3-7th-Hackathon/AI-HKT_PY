from youtube_transcript_api import YouTubeTranscriptApi

def load_youtube_script(video_id: str, lang="ko") -> str:
    transcript = YouTubeTranscriptApi.fetch(video_id, languages=[lang])
    return " ".join([s.text for s in transcript])


