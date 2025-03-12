import fasttext

class language_identification:
    def __init__(self):
        self.model = fasttext.load_model('cs336-data/cs336_data/model/lid.176.bin')

    def predict(self, text):
        text = text.replace("\n", "")
        result = self.model.predict(text)
        result = (result[0][0].replace('__label__', ''), float(result[1][0]))
        return result

if __name__=='__main__':
    langid = language_identification()
    for i in range(20):
        file_path = f'data/extract_warc{i+1}.txt'
        with open(file_path) as f:
            text = f.read()
            lan, scor = langid.predict(text)
            print("-"*20)
            print(f'Sample{i+1}')
            print(lan)
            print(scor)
            print("-"*20)
