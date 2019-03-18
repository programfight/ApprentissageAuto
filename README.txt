APPRENTISSAGE AUTO

-----------
| main.py | > Fichier à lancer pour l'apprentissage et la prédiction. Appelle le reste de nos fonctions dans tous les autres fichiers.
----------- > Respecte les spécifications pour le projet quant aux arguments d'entrée.

--------------------
| fit_n_predict.py | > Fonctions appellées par main.py.
-------------------- > 'Fit' entraîne sur les images dans le dossier spécifié et enregistre le classifieur dans 'Models'.
                     > 'Predict' le charge et lance la prédiction sur le dossier passé en argument à main.py.

-------------------
| output_stats.py | > Fonctions d'écriture dans un fichier pour stocker nos logs et garder nos résultats pour chaque algorithme.
------------------- > Ces résultats sont enregistrés au format .cvs pour ouverture facile par un tableur (excel, calc ...)

---------------
| imgmanip.py | > Fonctions de manipulation d'images pour ensuite pouvoir apprendre sur ces images manipulées.
---------------



 - DOSSIER ALGORITHMS :

-----------
| util.py | > Fonctions utilitaires rendant plus simples la lecture et l'écriture des fonctions d'apprentissage et de prédiction.
-----------

--------------
| current.py | > Fonctions utilitaires pour ne pas avoir à modifier beaucoup de code pour changer l'algorithme à utiliser.
--------------

----------------
| NB_simple.py | > Ensemble de fonctions d'apprentissage et de prédiction correspondant à l'algorithme Naive Bayes Gaussien
---------------- > différents paramètres (histogramme, histogramme normalisé, statistiques de l'image ...).



- DOSSIER MODELS :

--------------------
| saved_classifier | > Classifieur généré par la commande --fit. Est ensuite chargé par la commande --predict.
-------------------- > Généré grâce à joblib.
