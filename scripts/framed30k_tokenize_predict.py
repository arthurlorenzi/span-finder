import string
import pandas as pd
from argparse import ArgumentParser
from sftp import SpanPredictor
from spacy.lang.pt import Portuguese
from spacy.attrs import ORTH

# Create tokenizer
tokenizer = Portuguese().tokenizer

tokenizer.add_special_case("[NUM]", [{ ORTH: "[NUM]"}])
tokenizer.add_special_case("[NAME]", [{ ORTH: "[NAME]"}])
tokenizer.add_special_case("[DATE]", [{ ORTH: "[DATE]"}])
tokenizer.add_special_case("[TIME]", [{ ORTH: "[TIME]"}])
tokenizer.add_special_case("[VENUE]", [{ ORTH: "[VENUE]"}])
tokenizer.add_special_case("[TIMESPAN]", [{ ORTH: "[TIMESPAN]"}])
tokenizer.add_special_case("[...]", [{ ORTH: "[...]"}])

parser = ArgumentParser('predict spans for ICD data')
parser.add_argument('-m', help='model path', type=str)
parser.add_argument('-i', help='data path', type=str)
parser.add_argument('-o', help='output path', type=str)
args = parser.parse_args()

def tokenize(text):
	return list(map(str, tokenizer(text)))

# Load model
predictor = SpanPredictor.from_path(
	args.m,
	cuda_device=0,
)

with open(args.i) as f:
	lines = f.readlines()

tokens = list(map(lambda x: tokenize(x[1:-1]), lines)) # ignore quotes

output = predictor.predict_batch_sentences(
	tokens,
	output_format='json',
	progress=True
)

assert len(lines) == len(output)

df = pd.DataFrame({
	"tokens": tokens,
	"annotation": list(map(lambda x: x.span, output))
})

df.to_json(args.o, orient="records", lines=True)
