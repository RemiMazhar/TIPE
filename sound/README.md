# Explication de chaque script
## sendvalues.ino
Envoie sur le port série les valeurs lues par analogRead sur A0\
2 méthodes:
- envoi de texte (println(val)): ~2000Hz, mais l'IDE arduino est capable de recevoir les valeurs et les afficher en direct sur le serial monitor/plotter
- envoi en binaire: ~6000Hz, mais les données ne peuvent être lu que depuis arduinolistener.py
## arduinolistener.py
reçoit les données binaire envoyées par l'arduino et les enregistre dans un fichier .wav
## bandpass.py
simulation numérique d'un filtre passe bande (filtre passe bas et passe haut en série avec un suiveur entre les 2)\
Note: au final on va pas prendre ce filtre là pcq il est nul
## peakedetector.py
simulation numérique d'un détecteur de crête (aussi appelé détecteur d'enveloppe)
## full_circuit.py
applique les 2 filtres précédents à un fichier audio et enregistre le résultat
