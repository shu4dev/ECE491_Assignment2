from xopen import xopen
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding, bytes_to_str


def read_warc_file(file_path: str):
    with xopen(file_path, "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            yield record.reader.read()
        
def extract_text(html_bytes: bytes) -> str:
    html_str = bytes_to_str(html_bytes, detect_encoding(html_bytes))
    return extract_plain_text(html_str)

if __name__=='__main__':
    input_path = 'data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz'
    records = read_warc_file(input_path)
    
    # extract first 20 records and save as txt
    count = 0
    for record in tqdm(records):
        if count >= 20:
            break
        count += 1
        with open(f'data/extract_warc{count}.txt', 'w') as f:
            text = extract_text(record)
            f.write(text)
    print('Done')