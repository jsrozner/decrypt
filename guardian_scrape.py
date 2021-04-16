"""
For scraping the guardian cryptic crosswords.

# Notes on the Decrypting paper dataset
# 0 -> 10000: had 11 puzzles; omitted from dataset
# 10000-20000: not scraped
# 20000: 28259: done (present day as of the date we ran it)
"""
import argparse
import glob
import json
import logging
import os
import time
from collections import Counter
from collections import defaultdict
from typing import *
from typing import IO, Optional, Tuple, Dict

import urllib3
from bs4 import BeautifulSoup
from tqdm import tqdm

from decrypt.util import _gen_filename

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

min_size = 100 * 1024  # require that webpages are at least this big (otherwise no puzzle)
file_end = ".html"
BASE_URL = "https://www.theguardian.com/crosswords/"
k_subsite = "cryptic"
k_start_idx = 21465
k_end_idx = 28259


###
# Fetching methods
###
def _puzzle_json_from_file_or_str(input_data: Union[IO, str], ctr: Counter) -> Optional[Dict]:
    """
    Take a file handler, fh, representing HTML and return a dict from the parsed json
    Returns: Optional[Dict] if the html was successfully parsed
    """
    try:
        soup = BeautifulSoup(input_data, 'html.parser')
        cw_json_class = soup.findAll('div', {'class': 'js-crossword', 'data-crossword-data': True})
        cw_json_data = cw_json_class[0]['data-crossword-data']
    except Exception as e:
        print(e)
        ctr['html_fail'] += 1
        return None

    json_data = json.loads(cw_json_data)
    puzzle_entries = json_data.get('entries')

    if puzzle_entries is None:
        ctr['json_fail'] += 1
        return None

    return json_data


def _fetch(url, ctr: Counter, sleep_time=0.2, debug=False) -> Tuple[Optional[urllib3.HTTPResponse], int]:
    """
    Fetch from url, with retry if 429 response by sleeping in doubling increments of 0.2
    Returns: tuple:
                - resp if stat == 200, None otherwise
                - response code (-1 : failed to get any response, stat otherwise)
    """
    if debug:
        print(f"fetching {url}")
    if sleep_time > 10:  # base case
        print(f"sleeptime: {sleep_time} too large, skipping {url}")
        return None, -1

    resp = http.request("GET", url)
    stat = resp.status
    ctr[stat] += 1

    if resp.status == 429:  # rate limit
        time.sleep(sleep_time)
        _fetch(url, ctr, sleep_time * 2, debug=debug)  # recurse with higher wait

    if stat == 200:
        return resp, stat

    return None, stat  # failed, so just return the stat


def fetch_and_store_set(base_url: str,
                        subsite: str,
                        json_output_dir: str,
                        db: Dict[str, int],
                        start_idx: int,
                        stop_idx: int,
                        html_output_dir=None):
    """
    Fetch all puzzles in base_url/subsite/[start_idx-stop-idx] and write the puzzle json to output_dir
    If html_output_dir is not None, then also store a copy of the html (not just json)
    """

    def _fetch_and_store(idx: int) -> NoReturn:
        """
        Fetch from BASE_URL/subsite/idx and write the html to json_files_dir/<subsite><idx>.html
        """
        key = subsite + "/" + str(idx)
        url = base_url + key

        # this would require the database to be populated. will never trigger as currently written
        if db is not None and db.get(key) is not None:  # already checked url
            ctr['skipped: in db'] += 1
            return

        # check if file already exists
        json_outfile = _gen_filename(json_output_dir, subsite, ext=".json", idx=idx)
        if json_outfile in json_exists_set:
            ctr['skipped: json exists'] += 1
            return

        # otherwise proceed with _fetch
        resp, stat = _fetch(url, ctr)

        if resp is not None:  # stat == 200
            # parse the json
            data: str = resp.data.decode('utf-8')
            puz_json = _puzzle_json_from_file_or_str(data, ctr)
            if puz_json is None:
                print(f"error for {url}")
                return

            # write json if valid
            # json_outfile = _gen_filename(json_output_dir, subsite, ext=".json", idx=idx)
            with open(json_outfile, "w") as f:
                json.dump(puz_json, f)

            # write html files if required
            if html_output_dir is not None:
                outfile = _gen_filename(html_output_dir, subsite, ext=".html", idx=idx)
                with open(outfile, "w") as f:
                    print(data, file=f)

            ctr['success'] += 1
            ctr['totalclues'] += len(puz_json['entries'])  # we verified that entires is present

        # record status after finish processing
        if db is not None:
            db[key] = stat

    # first check for json that are already downloaded
    file_glob_path = _gen_filename(json_output_dir, subsite=k_subsite, ext=".json", return_glob=True)
    log.info(f'Using file glob at {file_glob_path}')
    file_glob = glob.glob(file_glob_path)
    log.info(f'Some files already present in the output directory; these will be skipped {len(file_glob)}')
    json_exists_set = set(file_glob)

    ctr = Counter()
    pbar = tqdm(range(start_idx, stop_idx + 1))
    for i in pbar:
        # pbar.set_description(f"Succ: {ctr[200]}\t Fail: {ctr[404]}\t Skip: {ctr['skipped: json exists']}\t")
        pbar.set_description(f"Succ: {ctr[200]}\t Fail: {ctr[404]}")
        _fetch_and_store(i)
    print(ctr)


def parse_args():
    parser = argparse.ArgumentParser('Guardian Scrape')

    parser.add_argument('--save_directory',
                        type=str,
                        required=True,
                        help='Where to save the downloaded json files')

    return parser.parse_args()


def main():
    #######################################
    # setup
    #######################################
    # keep track of what has already been fetched; useful if run is stopped
    # todo: this would need to be written / loaded to be useful; at present does nothing
    # instead we check whether a given json file exists; this works as long as the URL actually exists
    # for invalid URLs (i.e. invalid puzzle indices), we will retry download each time this is run
    url_db = defaultdict(None)

    # fetch website and store all json files
    # if you want the HTML, you can pass html_output_dir = html_output_dir
    log.info(f'Fetching puzzles from indexes {k_start_idx} to {k_end_idx} inclusive')
    fetch_and_store_set(BASE_URL,
                        subsite=k_subsite,
                        json_output_dir=parsed_args.save_directory,
                        db=url_db, start_idx=k_start_idx, stop_idx=k_end_idx, html_output_dir=None)


if __name__ == "__main__":
    parsed_args = parse_args()

    if not os.path.isdir(parsed_args.save_directory):
        raise NotImplemented(f'Save dir {parsed_args.save_directory} does not exist')

    http = urllib3.PoolManager()

    main()
