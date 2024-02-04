from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

tracknames = [] # List of tracks name
trackartists = [] # List of track artists

path = 'E:/Engineering/Signal Processing/Personal Projects/Breakbeat Analysis/Data/'
file_name = 'Ultimate_Breaks_and_Beats.csv'
full_file = path + file_name

base_url = "https://rateyourmusic.com/list/recordconvention/ultimate_breaks_and_beats/"
page_num = list(range(0,2))

for num in page_num:
    driver = webdriver.Chrome(ChromeDriverManager().install())
    
    url = base_url + str(num + 1)
    driver.get(url)
    content = driver.page_source
    soup = BeautifulSoup(content)

    table = soup.find('table', attrs={'id':'user_list'})
    body = table.find('tbody')
    for songs in body.findAll('tr', attrs={'class':['trodd', 'treven']}):
        main_entry = songs.find('td', attrs={'class':'main_entry'})
        
        try:
            h2 = main_entry.find('h2')
            track_artist = h2.find('a', attrs={'class':'list_artist'})
            track_artist = track_artist.text
    
            span = main_entry.find('span', attrs={'class':'hide-for-small-block'})
            trackname = span.find('span', attrs={'class':'rendered_text'})
            trackname = trackname.text
    
            tracknames.append(trackname)
            trackartists.append(track_artist)
        except:
            continue
    
    time.sleep(5)
        
df = pd.DataFrame({'Track Name':tracknames,'Track Artists':trackartists}) 
df.to_csv(full_file, index=False, encoding='utf-8')