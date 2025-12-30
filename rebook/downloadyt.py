import yt_dlp
from yt_dlp.utils import download_range_func

# Link zum YouTube Video
url = "https://www.youtube.com/watch?v=9Z17KL5HnwA"

start_time = 2  # accepts decimal value like 2.3
end_time = 7  

yt_opts = {
    'verbose': True,
    'download_ranges': download_range_func(None, [(start_time, end_time)]),
    'force_keyframes_at_cuts': True,
}

with yt_dlp.YoutubeDL(yt_opts) as ydl:
    ydl.download(url)