"""
Permet d'enregister les résultats d'apprentissage et de prédiction des algorithmes dans 1
fichier .cvs (séparateur ';').
"""

# Vide ou crée (s'il n'existe pas) un fichier .cvs.
# Par défaut fichier default_output.cvs
def clear_file(out_file = "./Stats/default_output"):
    path = out_file + ".cvs"
    open(path, "w")

# Ecrit une ligne dans "fichier", contenant 5 champs :
# Nom de l'algorithme, ses paramètres, le temps d'apprentissage,
# le temps de prédiction, et le score d'accuracy.
def add_stats(algoName, params, tpsApp, tpsPred, score, out_file = "./Stats/default_output"):
    path = out_file + ".cvs"
    stat_file = open(path, "a")

    tpsApp = str(tpsApp*1000)
    tpsPred = str(tpsPred*1000)

    score = str(score)

    line = algoName + ";" + params + ";" + tpsApp + ";" + tpsPred + ";" + score + "\n"
    stat_file.write(line)


# Vide ou crée un fichier et écrit immédiatement une ligne
# correspondant à une instance d'apprentissage.
def create_stats(algoName, params, tpsApp, tpsPred, score):
    clear_file()
    add_stats(algoName, params, tpsApp, tpsPred, score)
