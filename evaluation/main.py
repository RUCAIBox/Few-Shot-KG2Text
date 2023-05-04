import spacy
import json
from eval import Evaluate


class NLP:
    def __init__(self):
        self.nlp = spacy.load('./en_core_web_sm-2.3.1', disable=['ner', 'parser', 'tagger'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def sent_tokenize(self, text):
        doc = self.nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]
        return sentences

    def word_tokenize(self, text, lower=False):  # create a tokenizer function
        if text is None:
            return text
        text = ' '.join(text.split())
        if lower:
            text = text.lower()
        toks = [tok.text for tok in self.nlp.tokenizer(text)]
        return ' '.join(toks)


gold_file = "gold.txt"
reference_text = []
with open(gold_file, "r") as fin:
    for line in fin:
        text = line.strip()
        reference_text.append(text)

result_file = "generated.txt"
generated_text = []
with open(result_file, "r") as fin:
    for line in fin:
        text = line.strip()
        generated_text.append(text)

assert len(generated_text) == len(reference_text)
print(len(generated_text))

nlp = NLP()
generated_text = [nlp.word_tokenize(text, lower=True) for text in generated_text]
reference_text = [nlp.word_tokenize(text, lower=True) for text in reference_text]

calculator = Evaluate()
metric_dict = calculator.evaluate(generated_text, reference_text)
print(metric_dict)
