import fasttext

class toxicity:
    def __init__(self):
        self.model = fasttext.load_model('cs336-data/cs336_data/toxicity.ftz')