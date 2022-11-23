import os
from typing import Optional
import pandas as pd
import requests
from tqdm import tqdm


def download(url: str, name: str, chunk_size: int = 1024) -> None:
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(name, 'wb') as file, tqdm(
        desc=name,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def load_snap_reddithyperlinks_body(prefix: Optional[str] = "") -> pd.DataFrame:
    """
    load Network of subreddit-to-subreddit hyperlinks extracted from hyperlinks in the body of the post.
    https://snap.stanford.edu/data/soc-RedditHyperlinks.html
    :param prefix: (str) applies to path (before) for using from different entry points
    :return: pd.DataFrame
    """
    path = f'{prefix}data/raw/'
    name = f"{path}soc-redditHyperlinks-body.tsv"
    if not os.path.exists(name):
        url = "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"
        download(url, name)
    df = pd.read_csv(name, sep='\t')
    return df
