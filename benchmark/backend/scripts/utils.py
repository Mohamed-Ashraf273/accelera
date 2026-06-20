import re


def get_file_url(url):
    matchedID = re.search(r"/d/([^/]+)", url)
    if matchedID:
        file_id = matchedID.group(1)
        return f"https://drive.google.com/uc?id={file_id}"
    return url
