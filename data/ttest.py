import requests

url = "https://elastic.pushshift.io/rc/comments/_search"

querystring = {"source":'''{"query":{"bool":{"must":[{"simple_query_string":{"query":"Moving Ostwald borders back to the pre 1967 borders and declaring Palestinian statehood would remove the base for Hamas. Palestinians support Hamas because they're living in an occupied state, IDF are terrorists and they are known to randomly terrorize innocent civilians, invade their homes and steal their property. If the occupation was over, Palestine got their land back and they were now a recognized state Hamas would become irrelevant.  The argument that if they achieved statehood, they would get weapons and attack Israel is not only complete speculation, it's fairly irrelevant. It's like saying that we have to support a bully and help them keep this kid in a locker because if he gets out he might hit the bully, sometimes the right thing to do involves risk and Palestine getting better weapons to attack Israel ok a equal footing for the first time in 60 Years is a risk we have to take. ","fields":["body"],"default_operator":"and"}}],"filter":[{"range":{"created_utc":{"gte":null,"lte":null}}}],"should":[],"must_not":[]}},"size":100,"sort":{"created_utc":"desc"}}''', "source_content_type":"application/json"}

headers = {
    'Referer': "https://redditsearch.io/",
    'Cache-Control': "no-cache",
    'Postman-Token': "aef919bb-6795-9ab7-1447-e99d0aeea031",
    'Accept':"application/json",
    'content-type': 'application/json',
    'Origin': 'https://redditsearch.io',
    'Sec-Fetch-Mode': 'cors',
    'cookie':"__cfduid=d03bdea9eb6b86ac26408a1cf7cc82fb51571043217",
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
    }

# headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)