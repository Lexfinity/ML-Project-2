import pandas as pd
import requests

import requests
import urllib.parse
import json

URL = "https://elastic.pushshift.io/rc/comments/_search"

def get_subreddit(comment):
    query_string = create_querystring(comment)
    querystring = {"source": query_string, "source_content_type":"application/json"}
    headers = {
        'Referer': "https://redditsearch.io/",
        'accept':"*/*"
        }

    response = requests.request("GET", URL, headers=headers, params=querystring)

    return response.json()['hits']['hits'][0]['_source']['subreddit']

query_dict = {
   "query":{
      "bool":{
         "must":[
            {
               "simple_query_string":{
                  "query":"You have to take it out of the bag, Morty!",
                  "fields":[
                     "body"
                  ],
                  "default_operator":"and"
               }
            }
         ],
         "filter":[
            {
               "range":{
                  "created_utc":{
                     "gte":0,
                     "lte":1571045159
                  }
               }
            },
            {
               "terms":{
                  "subreddit":[
                     "hockey",
                     "nba",
                     "leagueoflegends",
                     "soccer",
                     "funny",
                     "movies",
                     "anime",
                     "overwatch",
                     "trees",
                     "globaloffensive",
                     "nfl",
                     "askreddit",
                     "gameofthrones",
                     "conspiracy",
                     "worldnews",
                     "wow",
                     "europe",
                     "canada",
                     "music",
                     "baseball"
                  ]
               }
            }
         ],
         "should":[

         ],
         "must_not":[

         ]
      }
   },
   "size":100,
   "sort":{
      "created_utc":"desc"
   }
}
def create_querystring(comment):
    # query_dict = {"query": {"bool": {"must": [{"simple_query_string":{"query": "somthing", "fields":["body"], "default_operator": "and"} }], "filter": [{ "range": {"created_utc": {"gte": None, "lte": None} } }], "should":[], "must_not":[] }, "size":100, "sort": {"created_utc": "desc"}}
    query_dict['query']['bool']['must'][0]['simple_query_string']['query'] = comment
    return json.dumps(query_dict)
   

if __name__ == "__main__":

    # print(get_subreddit('ss'))
    df = pd.read_csv("reddit_test.csv")
    index_sf = 0

    df['subreddits']=pd.Series([])
    for index, row in df[index_sf:].iterrows():
        try:
            subreddit = get_subreddit(row.comments)
            df.loc[index, 'subreddits'] = subreddit
            print('index {} done', index, subreddit)
            df.to_csv("test.csv")
        except:
            df.loc[index, 'subreddits'] = ''
            print('index {} had errors', index, subreddit)
            df.to_csv("test.csv")
        # df.loc[index, 'subreddits']='a'    


    

