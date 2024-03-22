import string
import pandas as pd
from argparse import ArgumentParser
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

parser = ArgumentParser('tokenize data for inference')
parser.add_argument('-i', help='input path', type=str)
parser.add_argument('-o', help='output path', type=str)
args = parser.parse_args()

# Preprocessing & tokenization functions
def preprocess(text):
	text = text.replace('\r\n', '\n')

	return [
		s if s[-1] in string.punctuation else s + '.'
		for s in text.split('\n')
		if s.strip() != ''
	]

def tokenize(text):
	return list(map(str, tokenizer(text)))

## Read data 
dtypes = {
	"id": int,
	"tbes_ds_subjetivo": str,
	"tbeo_ds_objetivo": str,
	"tbea_ds_avaliacao": str,
	"tbep_ds_plano": str,
	"tbenc_ds_complemento": str,
	"tbenc_ds_motivo_encaminhamento": str,
	"tbenc_ds_observacao": str,
	"ds_obs": str,
}
df = pd.read_csv(args.i, dtype=dtypes, usecols=list(dtypes.keys()), index_col="id")

# Preprocess and tokenize
tokenized = df.stack().map(preprocess).explode().dropna().map(tokenize)
output = tokenized.groupby(level=[0,1]).agg(list).unstack().reindex(df.index)

assert len(df) == len(output)

# Save tokenized data
output.to_json(args.o, orient="records", lines=True)
