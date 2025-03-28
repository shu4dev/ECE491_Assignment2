import fasttext
from huggingface_hub import hf_hub_download

nsfw_model = hf_hub_download(repo_id="shu4dev/nsfw", filename="jigsaw_fasttext_bigrams_nsfw_final.bin")
toxic_model = hf_hub_download(repo_id="shu4dev/toxic", filename="jigsaw_fasttext_bigrams_hatespeech_final.bin")
class toxicity:
    def __init__(self):
        self.nsfw_model = fasttext.load_model(nsfw_model)
        self.toxic_model = fasttext.load_model(toxic_model)
    def classify_nsfw(self, text):
        text = text.replace("\n", "")
        result = self.nsfw_model.predict(text)
        return (result[0][0].replace('__label__', ''), float(result[1][0]))

    def classify_toxic(self, text):
        text = text.replace("\n", "")
        result =  self.toxic_model.predict(text)
        return (result[0][0].replace('__label__', ''), float(result[1][0]))

if __name__=='__main__':
    model = toxicity()
    for i in range(20):
        file_path = f'data/extract_warc{i+1}.txt'
        with open(file_path) as f:
            text = f.read()
            nsfw, score1 = model.classify_nsfw(text)
            tox, score2  = model.classify_toxic(text)
            print("-"*20)
            print(f'Sample{i+1}')
            print(nsfw, score1)
            print(tox, score2)
            print("-"*20)