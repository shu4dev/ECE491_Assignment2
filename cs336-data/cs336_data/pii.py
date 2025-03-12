import regex as re

class pii:
    def mask_emails(self, text:str)-> tuple[str, int]:
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        replace = "|||EMAIL_ADDRESS|||"
        result, count = re.subn(pattern, replace, text)
        return result, count

    def mask_phone_numbers(self, text)-> tuple[str, int]:
        pattern = r'\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}'
        replace = "|||PHONE_NUMBER|||"
        result, count = re.subn(pattern, replace, text)
        return result, count
    
    def mask_ips(self, text)-> tuple[str, int]:
        pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        replace = "|||IP_ADDRESS|||"
        result, count = re.subn(pattern, replace, text)
        return result, count

if __name__ == "__main__":
    p = pii()
    for i in range(0, 10):
        file_path = f'data/extract_warc{i+1}.txt'
        with open(file_path) as f:
            text = f.read()
            
            print("-"*20)
            print(f'Sample{i+1}')
            masked_text, count = p.mask_emails(text)
            masked_text, count = p.mask_phone_numbers(masked_text)
            masked_text, count = p.mask_ips(masked_text)
            print(masked_text)
            print("-"*20)