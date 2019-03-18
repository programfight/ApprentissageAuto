import sys
from fit_n_predict import fit
from fit_n_predict import predict
from fit_n_predict import calc_score

def usage(filecode):
    print("Entrainement : " + filecode + ": --fit     <folder>")
    print("Prédiction :   " + filecode + ": --predict <folder>")

if len(sys.argv) > 2:
	filecode,typeof,folder=sys.argv
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
