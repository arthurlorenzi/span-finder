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

## Read data and make 
dtypes = {
	"icd_code": str,
	"icd_name_fine": str,
	"icd_name_middle": str,
	"icd_name_coarse": str,
}

df = pd.read_json(args.i, orient="records", lines=True, dtype=dtypes)
tokens = df["icd_name_fine"].map(tokenize)

output = predictor.predict_batch_sentences(
	tokens.tolist(),
	output_format='json',
	progress=True
)

assert len(df) == len(output)

df["tokens"] = tokens
df["annotation"] = pd.Series(list(map(lambda x: x.span, output)), df.index)

df.to_json(args.o, orient="records", lines=True)
