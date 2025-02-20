import yt_dlp
import ffmpeg
import numpy as np
import librosa
import soundfile as sf
import io
import pandas as pd
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from audio_process import Audio_Process

# Semaphore to limit simultaneous YouTube requests (avoid rate limiting)
semaphore = threading.Semaphore(5)  # Adjust this based on network speed

# Set up logging to track errors
logging.basicConfig(filename="errors.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

def search_youtube(song_name, artist_name=None):
    """
    Search YouTube for a song and return its best audio URL, with error handling
    """
    try:
        with semaphore: # Limit concurrent requests
            query = f"{artist_name} - {song_name} official audio" if artist_name else song_name
            search_url = f"ytsearch: {query}"

            ydl_opts = {"format": "bestaudio/best", "quiet": True}

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(search_url, download=False)

            if "entries" in result and result["entries"]:
                return result["entries"][0]["url"]  # Return first result
            else:
                logging.error(f"No results found for: {query}")
                return None
    except Exception as e:
        logging.error(f"Error searching YouTube for {query}: {str(e)}")
        return None  # Gracefully skip errors

def stream_youtube_audio(url):
    """
    Stream YouTube audio and return waveform data, with retries.
    """
    try:
        process = (
            ffmpeg.input(url)
            .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=44100)
            .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
        )

        audio_data, _ = process.communicate()
        if not audio_data:
            logging.error(f"Failed to download audio from: {url}")
            return None, None

        audio_buffer = io.BytesIO(audio_data)

        # Read as WAV file and process with librosa
        y, sr = sf.read(audio_buffer, dtype='float32')

        return y, sr
    except Exception as e:
        logging.error(f"Error processing audio from {url}: {str(e)}")
        return None, None  # Return empty values to prevent crashes

def search_youtube_with_retry(song_name, artist_name=None, retries=3, delay=5):
    """Retries YouTube search if it fails."""
    for attempt in range(retries):
        url = search_youtube(song_name, artist_name)
        if url:
            return url  # Success!
        logging.warning(f"Retry {attempt+1}/{retries} for: {song_name} by {artist_name}")
        time.sleep(delay)  # Wait before retrying

    logging.error(f"Failed to retrieve URL after {retries} attempts: {song_name}")
    return None  # Return None if all retries fail

def process_song(row):
    """Process a single song from CSV (search, download, extract features), with error handling."""
    try:
        song_dict = {}
        data_dict = {}

        song_name = row["Track Name"]
        artist_name = row["Track Artists"]
        key = f"{artist_name} - {song_name}"

        song_url = search_youtube_with_retry(song_name, artist_name)

        if not song_url:
            logging.error(f"Skipping {key}: No valid YouTube URL found.")
            return None  # Skip if no valid URL

        y, sr = stream_youtube_audio(song_url)
        if y is None or sr is None:
            logging.error(f"Skipping {key}: Audio processing failed.")
            return None  # Skip if audio processing failed

        song_processed_data = Audio_Process(y, sr)

        data_dict["Time Domain"] = (y, sr)
        data_dict["Frequency Domain"] = song_processed_data
        song_dict[key] = data_dict
    except Exception as e:
        logging.error(f"Unexpected error processing {key}: {str(e)}")
        return None  # Gracefully handle unexpected failures

# Read CSV
csv_file = "E:/Engineering/Signal Processing/Personal Projects/Breakbeat Analysis/Data/Breakbeats_Master_List.csv"
csv_df = pd.read_csv(csv_file)

# Multithreading: Process multiple songs in parallel
audio_process_list = []

with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_song = {executor.submit(process_song, row): row for _, row in csv_df.iterrows()}

    for future in as_completed(future_to_song):
        result = future.result()
        if result:  # Only store valid results
            audio_process_list.append(result)

# Save results to JSON
json_file = "E:/Engineering/Signal Processing/Personal Projects/Breakbeat Analysis/Data/breakbeat_data.json"
with open(json_file, "w") as file:
    import json
    json.dump(audio_process_list, file, indent=4)

print(f"Processing completed! {len(audio_process_list)} valid samples saved.")