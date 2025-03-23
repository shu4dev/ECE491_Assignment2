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
    input_path = 'subsampled_positive_urls.warc.gz'
    records = read_warc_file(input_path)
    
    count = 0
    for record in tqdm(records):
        count += 1
        with open(f'./positive_sample/sample_{count}.txt', 'w') as f:
            text = extract_text(record)
            f.write(text)
    print('Done')