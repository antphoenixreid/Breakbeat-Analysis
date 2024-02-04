import pandas as pd
from pytube import YouTube, Search
from moviepy.editor import *

import os
import glob

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Initial Variables
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="1d96de8cf09f4ff89a0189d2c864e6c0",
                                               client_secret="a0b53ad8827b4505ac85a93cf7d7f523",
                                               redirect_uri="http://127.0.0.1:9090",
                                               scope="user-library-read"))

header = ['Artist', 'Track Title', 'Duration (minutes)', 'Key', 'Time Signature',
          'Tempo', 'Energy', 'Danceability', 'Speechiness', 'Acousticness',
          'Instrumentalness', 'Liveness', 'Valence']

pitch_class = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

path = 'E:/Engineering/Signal Processing/Personal Projects/Breakbeat Analysis/Data/'
file_name = 'Most_Sampled_Breakbeat.xlsx'
full_file = path + file_name

csv = pd.read_excel(full_file, sheet_name=['Most_Sampled_Breakbeat','Bboy_Dojo_Breakbeat', 'Ultimate_Breakbeat'])

# Get DataFrame
list_1 = csv.get('Most_Sampled_Breakbeat')
list_1 = list_1[['Song', 'Artist']].values.tolist()
list_1 = [tuple(track) for track in list_1]

list_2 = csv.get('Bboy_Dojo_Breakbeat')
list_2 = list_2[['Song', 'Artist']].values.tolist()
list_2 = [tuple(track) for track in list_2]

list_3 = csv.get('Ultimate_Breakbeat')
list_3 = list_3[['Song', 'Artist']].values.tolist()
list_3 = [tuple(track) for track in list_3]

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def MP4ToMP3(mp4, mp3):
    FILETOCONVERT = AudioFileClip(mp4)
    FILETOCONVERT.write_audiofile(mp3)
    FILETOCONVERT.close()
    
# Function: extract_meta_data
def extract_meta_data(artist, track):
    results = sp.search(q="artist:" + artist + " track:" + track, type="track")

# List of intersecting Beats - Bboy beats with the most samples (most data to work with)
# inter_tracks = intersection(most_sampled_tracks,songs_with_breakbeats)
combined_list = list(set(list_1 + list_2))
combined_list = list(set(combined_list + list_3))

track_name_list = []
track_artist_list = []

# Download the Songs
for i in range(len(combined_list)):
    # Get the Keywords
    track_name = combined_list[i][0]
    if isinstance(track_name, int):
        track_name = str(track_name)
        
    track_name_list.append(track_name)
        
    track_artist = combined_list[i][1]
    if isinstance(track_artist, int):
        track_artist = str(track_artist)
        
    track_artist_list.append(track_artist)
    
    track_info = track_name + ' ' + track_artist
    
    # Search YouTube
    # yt_search_id = Search(track_info).results[0].video_id
    s = Search(track_info)
    
    if track_info == 'Groove To Get Down T-Connection':
        video_id = '1JpYhHy65tc'
        yt_link = "https://www.youtube.com/watch?v=" + video_id
    elif track_info == 'UFO ESG':
        video_id = 'iAH9yD4Sxqo'
        yt_link = "https://www.youtube.com/watch?v=" + video_id
    else:
        for k in range(len(s.results)):
            video_id = s.results[k].video_id
            temp_song_yt_link = "https://www.youtube.com/watch?v=" + video_id
            yt_track_title = YouTube(temp_song_yt_link).streams[0].title
            
            if yt_track_title.find(track_name) != -1 or yt_track_title.find(track_artist) != -1 and yt_track_title.find('Instrumental') == -1:
                yt_link = temp_song_yt_link
                break
    
    # Create Directory and Download
    song_folder = track_name + '-' + track_artist
    song_folder = song_folder.replace('"','')
    song_folder = song_folder.replace("'","")
    song_folder = song_folder.replace(":","")
    song_folder = song_folder.replace(";","")
    song_folder = song_folder.replace(",","")
    song_folder = song_folder.replace("#","")
    song_folder = song_folder.replace("&","")
    song_folder = song_folder.replace("{","")
    song_folder = song_folder.replace("}","")
    song_folder = song_folder.replace("<","")
    song_folder = song_folder.replace(">","")
    song_folder = song_folder.replace("*","")
    song_folder = song_folder.replace("?","")
    song_folder = song_folder.replace("!","")
    song_folder = song_folder.replace("@","")
    song_folder = song_folder.replace("+","")
    song_folder = song_folder.replace("=","")
    song_folder = song_folder.replace("|","")
    song_folder = song_folder.replace("$","")
    song_folder = song_folder.replace(".","")
    
    song_path = path + song_folder + '/'
    
    if not os.path.exists(song_path):
        os.mkdir(song_path)
    else:
        files = glob.glob(song_path + '*')
        for f in files:
            os.remove(f)
    
    selected_video = YouTube(yt_link,
                             use_oauth=True,
                             allow_oauth_cache=True)
    
    audio = selected_video.streams.filter(only_audio=True).first()
    audio.download(output_path=song_path)
    
    # Check for the MP4 file, convert it to MP3, Delete MP4 file
    mp4_file = glob.glob(os.path.join(song_path, '*.mp4'))[0]
    mp3_file = os.path.splitext(mp4_file)[0] + '.mp3'
    
    MP4ToMP3(mp4_file, mp3_file)
    
    os.remove(mp4_file)

    # Search in Spotify
    searchResults = sp.search(q="artist:" + track_artist + " track:" + track_name, type="track")

    track_info = searchResults['tracks']['items']
    if track_info:
        track_info = track_info[0]
        sp_track_info = []

        sp_track_name = track_info['name']
        sp_track_info.append(sp_track_name)
        sp_track_artist = track_info['artists'][0]['name']
        sp_track_info.append(sp_track_artist)
        sp_track_duration = track_info['duration_ms']/60000
        sp_track_info.append(sp_track_duration)

        sp_track_meta = sp.audio_features(track_info['uri'])[0]

        sp_track_key = pitch_class[sp_track_meta['key']]
        sp_track_info.append(sp_track_key)
        sp_track_time_sign = sp_track_meta['time_signature']
        sp_track_info.append(sp_track_time_sign)
        sp_track_tempo = sp_track_meta['tempo']
        sp_track_info.append(sp_track_tempo)
        sp_track_energy = sp_track_meta['energy']
        sp_track_info.append(sp_track_energy)
        sp_track_danceability = sp_track_meta['danceability']
        sp_track_info.append(sp_track_danceability)
        sp_track_speechiness = sp_track_meta['speechiness']
        sp_track_info.append(sp_track_speechiness)
        sp_track_acousticness = sp_track_meta['acousticness']
        sp_track_info.append(sp_track_acousticness)
        sp_track_instrumentalness = sp_track_meta['instrumentalness']
        sp_track_info.append(sp_track_instrumentalness)
        sp_track_liveness = sp_track_meta['liveness']
        sp_track_info.append(sp_track_liveness)
        sp_track_valence = sp_track_meta['valence']
        sp_track_info.append(sp_track_valence)

        text_file_name = track_name + "_" + track_artist + '.txt'
        text_file_name = text_file_name.replace('"','')
        text_file_name = text_file_name.replace("'","")
        text_file_name = text_file_name.replace(":","")
        text_file_name = text_file_name.replace(";","")
        text_file_name = text_file_name.replace(",","")
        text_file_name = text_file_name.replace("#","")
        text_file_name = text_file_name.replace("&","")
        text_file_name = text_file_name.replace("{","")
        text_file_name = text_file_name.replace("}","")
        text_file_name = text_file_name.replace("<","")
        text_file_name = text_file_name.replace(">","")
        text_file_name = text_file_name.replace("*","")
        text_file_name = text_file_name.replace("?","")
        text_file_name = text_file_name.replace("!","")
        text_file_name = text_file_name.replace("@","")
        text_file_name = text_file_name.replace("+","")
        text_file_name = text_file_name.replace("=","")
        text_file_name = text_file_name.replace("|","")
        text_file_name = text_file_name.replace("$","")

        text_file_path = song_path + text_file_name

        f = open(text_file_path, 'a')
        for i in range(len(header)):
            f.write(header[i] + ': ' + str(sp_track_info[i]) + '\n')

        f.close()
        
# Create the Master List of Breakbeats
file_name = 'Breakbeats_Master_List.csv'
full_file = path + file_name

df = pd.DataFrame({'Track Name':track_name_list,'Track Artists':track_artist_list}) 
df.to_csv(full_file, index=False, encoding='utf-8')