from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

tracknames = [] # List of tracks name
trackartists = [] # List of track artists
sample_count = [] # List of sample counts for tracks

base_url = "https://www.whosampled.com/most-sampled-tracks/"
page_num = list(range(1,11))

for num in page_num:
    driver = webdriver.Chrome(ChromeDriverManager().install())
    
    url = base_url + str(num)
    driver.get(url)
    content = driver.page_source
    soup = BeautifulSoup(content)
    
    for li in soup.findAll('li', attrs={'class':'listEntry trackCompactEntry'}):
        track = li.find('span', attrs={'class':'trackName'})
        track = track.text
        track = track.split(' - ')
        trackname = track[0]
        trackartist = track[1]
        
        count = li.find('span', attrs={'class':'counts'})
        count = count.text
        count = count.replace('Sampled ', '')
        count = count.replace(' times', '')
        
        tracknames.append(trackname)
        trackartists.append(trackartist)
        sample_count.append(count)
        
    # time.sleep(5)
        
df = pd.DataFrame({'Track Name':tracknames,'Track Artists':trackartists,'Number of Tracks that Sampled':sample_count}) 
df.to_csv('Most_Sampled_Tracks.csv', index=False, encoding='utf-8')