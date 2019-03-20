import sys
from fit_n_predict import fit
from fit_n_predict import predict
from fit_n_predict import calc_score
from fit_n_predict import score_split

def usage(filecode):
    print("Entrainement : " + filecode + ": --fit                       <folder>")
    print("Prédiction :   " + filecode + ": --predict                   <folder>")
    print("Score :        " + filecode + ": --score                     <folder>")
    print("Split & Score :" + filecode + ": --split [split_percent]     <folder>")

if len(sys.argv) > 3:
    filecode, typeof, test_percent, folder = sys.argv
    if typeof == "--split":
        score_split(folder, float(test_percent))
    else:
        usage(filecode)
elif len(sys.argv) > 2:
	filecode, typeof, folder = sys.argv
	if typeof == "--fit":
		fit(folder)
	elif typeof == "--predict":
		predict(folder)
	elif typeof == "--score":
		calc_score(folder)
	else:
		usage(filecode)

else:
    filecode = sys.argv[0]
    usage(filecode)
