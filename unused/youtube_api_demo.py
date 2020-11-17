
# pip install virtualenv
# virtualenv cs229-project
# source cs229-project/bin/activate
# <your-env>/bin/pip install google-api-python-client
# cs229-project/bin/pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

import os
from googleapiclient.discovery import build

api_key = "insert your API Key"

youtube = build('youtube', 'v3', developerKey=api_key)

request = youtube.channels().list(
    part = 'contentDetails, statistics',
    forUsername = 'schafer5'
)

# pl_request = youtube.playlists().list(
#     part = 'contentDetails, snippet',
#     channelId = "UCCezIgC97PvUuR4_gbFUs5g"
# )

# pl_request = youtube.playlistItems().list(
#     part = 'contentDetails',
#     playlistId = "PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS"
# )

response = request.execute()
#pl_response = pl_request.execute()

for item in response['items']:
    print(item)
    print("\n")

print(pl_response)

# Kids, beauty, music, art, movies, video_views --> subscriber_counts
# category tags, 
