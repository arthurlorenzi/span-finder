import pandas as pd
from argparse import ArgumentParser
from sftp import SpanPredictor

parser = ArgumentParser('predict spans for public health data')
parser.add_argument('-m', help='model path', type=str)
parser.add_argument('-i', help='data path', type=str)
parser.add_argument('-o', help='output path', type=str)
args = parser.parse_args()

# Load model
predictor = SpanPredictor.from_path(
	args.m,
	cuda_device=-1,
)

# Open output file
fp = open(args.o, 'w')

## Read data and make 
dtypes = {
	"id": int,
	"tbes_ds_subjetivo": object,
	"tbeo_ds_objetivo": object,
	"tbea_ds_avaliacao": object,
	"tbep_ds_plano": object,
	"tbenc_ds_complemento": object,
	"tbenc_ds_motivo_encaminhamento": object,
	"tbenc_ds_observacao": object,
	"ds_obs": object,
}

for df in pd.read_json(args.i, orient="records", lines=True, dtype=dtypes, chunksize=100000):
	sents = df.stack().dropna().explode()

	output = predictor.predict_batch_sentences(
		sents.tolist(),
		output_format='json',
		progress=True
	)

	output = pd.Series(map(lambda x: (x.span, x.sentence), output), sents.index)
	output = output.groupby(level=[0,1]).agg(list).unstack().reindex(df.index)

	assert len(df) == len(output)

	# Save data
	for _, row in output.iterrows():
		row = row[~row.isnull()]
		if len(row) > 0:
			fp.write(row.to_json())
			fp.write('\n')
		else:
			fp.write('{}\n')

# Close output file
fp.close()