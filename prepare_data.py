import argparse
import requests
import tarfile
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data downloading")
    parser.add_argument('--dataset', choices=["synthetic", "yahoo", "yelp", "omniglot", "all"], 
        default="all", help='dataset to use')

    args = parser.parse_args()

    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    os.chdir("datasets")

    synthetic_id = "1pEHLedf3ZSo7UrHdvR1VWPfWNTcN6oWH"
    yahoo_id = "13azGlTuGdzWLCmgDmQPmvb_jcexVWX7i"
    yelp_id = "1FT49oLNV8syhmGXEgiK6XTjEfMNqqEJJ"
    omniglot_id = "1PbFAm2wtXfdnV7ZixImkBubC0A6-XwHU"

    if args.dataset == "synthetic":
        file_id = [synthetic_id]
    elif args.dataset == "yahoo":
        file_id = [yahoo_id]
    elif args.dataset == "yelp":
        file_id = [yelp_id]
    elif args.dataset == "omniglot":
        file_id = [omniglot_id]
    else:
        file_id = [synthetic_id, yahoo_id, yelp_id, omniglot_id]

    destination = "datasets.tar.gz"

    for file_id_e in file_id:
        download_file_from_google_drive(file_id_e, destination)  
        tar = tarfile.open(destination, "r:gz")
        tar.extractall()
        tar.close()
        os.remove(destination)

    os.chdir("../")

