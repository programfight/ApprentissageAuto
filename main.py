import sys
from fit_n_predict import fit
from fit_n_predict import predict
from fit_n_predict import calc_score
from fit_n_predict import score_split

def usage(filecode):
    print("\nUsage :\n\nEntrainement : " + filecode + ": --fit                           <folder>")
    print("Prédiction :   " + filecode + ": --predict                       <folder>")
    print("Score :        " + filecode + ": --score [type score]            <folder>")
    print("Split & Score :" + filecode + ": --split [% test] [type score]   <folder>\n")

if len(sys.argv) > 4:
    filecode, typeof, split_percentage, score_type, folder = sys.argv
    if typeof == "--split":
        score_split(folder, float(split_percentage), score_type)
elif len(sys.argv) > 3:
    filecode, typeof, score_type, folder = sys.argv
    if typeof == "--score":
    	calc_score(folder, score_type)
    else:
        usage(filecode)
elif len(sys.argv) > 2:
	filecode, typeof, folder = sys.argv
	if typeof == "--fit":
		fit(folder)
	elif typeof == "--predict":
		predict(folder)
	else:
		usage(filecode)

else:
    filecode = sys.argv[0]
    usage(filecode)
