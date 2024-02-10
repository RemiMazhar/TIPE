# Description du processus
https://docs.google.com/document/d/1LRA2W9hHjYT-9NBD--So-HSOaEKgGpTQN6bspH7mPEk/edit?usp=sharing
## Détection de la position de la caméra
### Détection de la table sur l’image
#### Détection des triangles
- Racisme (discriminer les pixels en fonction de la couleur)
- Détection de composantes connexes
- Triangle circonscrit de l’enveloppe convexe puis rapport des aires
#### Numérotation des triangles
- hypothèse: la caméra est au-dessus de la table
- déterminants pour les numéroter dans le bon ordre
### Résolution numérique pour la position de la caméra
- Fonction à minimiser / système d’équations
- Algo pour la minimiser
- Solutions parasites (et comment on les élimine)
### Résolution explicite pour la position de la caméra
- [cf papier](https://perception.inrialpes.fr/Publications/1999/QL99/Quan-pami99.pdf)
## Détection de la position de la balle
### Détection de la balle sur l’image
- Racisme (discriminer les pixels en fonction de la couleur)
- Détection de composantes connexes
- Discrimination score de roundness-fatness
  - Best fit circle (least squares) de l’enveloppe convexe puis rapport des aires
### Positionnement
- Calcul de la demi-droite à partir de la position sur l’image (xN images)
- Descente de gradient pour intersection
## Applications / expériences
### Mesure de g
- Descente de gradient pour fit une parabole

# Explication de chaque script
## calibration.py
- en commentaire: définition du repère associé à la caméra et du rapport RATIO
- script pour mesurer le RATIO d'une caméra à partir d'une image
## geometry.py
- définitions des classes Droite, Sphere et Plan (seule Droite est utilisée au final), des objets géométriques auxquels on peut calculer la distance et le gradient de cette distance^2, utilisés pour la descente de gradient lors du positionnement
- définition de la classe Camera: une caméra est définie par sa position, son orientation (une matrice), et son RATIO (voir calibration.py pour définition)
## object_detection.py
toutes les méthodes liées à la détection d'objets sur une image:
- détection de ronds oranges pour une balle
- détection et numérotation de triangles colorés pour la table
## locationOOP.py
utilise tous les scripts précédents pour trouver la position de la balle à partir de N images chacune associée à la caméra qui l'a prise
## locate_camera.py
calcule numériquement la position de la caméra à partir de points repérés sur l'image dont on connait la position réelle (dans l'espace)\
contient également une petite UI pour cliquer sur ces points de référence manuellement
## PnP.py
fait la même chose avec la méthode explicite de Quan et Lan au lieu d'une méthode numérique\
les formules sont précalculées (ie calculée à l'avance et stockée) sous forme de polynomes en plein de variables, qu'il faut évaluer lors de l'exécution
## monom.npy
array numpy représentant tous les monomes des formules susmentionnées
## polynom_evaluation.pyh
différentes expériences pour accélérer l'évaluation des formules de PnP\
essentiellement l'idée est de trouver la manière optimale de calculer tous les monomes (ex: pour calculer x**8y**9 et x**9y**8, on pourra calculer d'abord (xy)**8, puis multiplier par x ou par y), puis enregistrer cet ordre de calcul, et le suivre lors de l'exécution
## video_formatter.py
mini UI pour préparer une vidéo à être analysée (sélectionner une portion de la vidéo, crop si nécessaire)\
en théorie pourrait être fait avec n'importe quel truc pour edit des vidéos mais souvent ces logiciels supportent mal les hautes framerates
## fitparabola.py
le nom est relativement explicite pour une fois
## video_analyzer.py
- get_trajectory: applique locationOOP.py à chaque frame d'une vidéo
- expériences de mesure de g
- dans le main: exécute presque tout le processus (sauf la détection des triangles car pas encore assez fiable) en utilisant locate_camera.py et locateOOP.py
