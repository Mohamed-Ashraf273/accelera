from accelera.src.utils.dataset_retriever import retriever
import json
import sys
import re


def normalize_drive_url(url):
    matchedID = re.search(r"/d/([^/]+)", url)
    if matchedID:
        file_id = matchedID.group(1)
        return f"https://drive.google.com/uc?id={file_id}"
    return url
drive_file_link=sys.argv[1]
target_column_name=sys.argv[2]
user_id=sys.argv[3]
try:
    url=normalize_drive_url(drive_file_link)
    retriever.connect()
    df = retriever.retrieve_dataset(f"{user_id}",url=url, df=True)
    if target_column_name not in df.columns:
       print(json.dumps({"message":"Dataset dosent have this target column","isValid":False}))
       sys.exit(0)
    print(json.dumps({"message":"Dataset has this target column","isValid":True}))

    sys.exit(0)

except Exception as e:
    print(json.dumps({"message":str(e),"isValid":False}))
    sys.exit(1)
finally:
    try:
        retriever.close()
    except:
        pass