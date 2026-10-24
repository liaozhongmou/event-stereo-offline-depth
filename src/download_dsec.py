"""Download one explicitly selected official DSEC sequence and hash its files."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import argparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/dsec/thun_00_a'
BASE = 'https://download.ifi.uzh.ch/rpg/DSEC/train/thun_00_a/thun_00_a_'
FILES = ['calibration.zip', 'events_left.zip', 'events_right.zip',
         'disparity_event.zip', 'disparity_timestamps.txt']


def fetch(name):
    path = OUT / name
    url = BASE + name
    with urllib.request.urlopen(url, timeout=90) as response, path.open('xb') as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            target=OUT/name.removesuffix('.zip') if name.startswith('events_') else OUT
            target.mkdir(parents=True,exist_ok=True)
            archive.extractall(target)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = dict(file=name, url=url, bytes=path.stat().st_size, sha256=digest)
    print(json.dumps(result), flush=True)
    return result


def ranged(name):
    """Bounded parallel ranges for the two large archives; validate every range."""
    path = OUT / name
    url = BASE + name
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as response:
        size = int(response.headers['Content-Length'])
    block = 4*1024*1024
    def part(start):
        end = min(start+block,size)-1
        request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request,timeout=90) as response:
            assert response.status==206
            assert response.headers['Content-Range']==f'bytes {start}-{end}/{size}'
            data=response.read()
        assert len(data)==end-start+1
        return data
    offset=path.stat().st_size
    assert offset==size or offset%block==0, 'Resume only verified complete range prefixes'
    with ThreadPoolExecutor(max_workers=6) as pool, path.open('ab') as stream:
        for index,data in enumerate(pool.map(part,range(offset,size,block))):
            stream.write(data)
            if index%10==0:
                print(name,stream.tell(),size,flush=True)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        target=OUT/name.removesuffix('.zip')
        target.mkdir(exist_ok=False)
        archive.extractall(target)
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(file=name,url=url,bytes=size,sha256=digest)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--finish-ranges',action='store_true')
    args=parser.parse_args()
    if args.finish_ranges:
        records=[]
        for name in FILES:
            if name in ['events_left.zip','events_right.zip']:
                record=ranged(name)
            else:
                path=OUT/name
                with path.open('rb') as stream:
                    digest=hashlib.file_digest(stream,'sha256').hexdigest()
                record=dict(file=name,url=BASE+name,bytes=path.stat().st_size,sha256=digest)
            records.append(record)
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        with ThreadPoolExecutor(max_workers=3) as pool:
            records = list(pool.map(fetch, FILES))
    (OUT / 'download_manifest.json').write_text(json.dumps(dict(
        dataset='DSEC', sequence='thun_00_a', license='CC BY-SA 4.0',
        source='https://dsec.ifi.uzh.ch/dsec-datasets/download/', files=records), indent=2),
        encoding='utf-8')
