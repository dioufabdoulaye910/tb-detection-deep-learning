# tb-detection-deep-learning
# Description du projet

Ce projet vise à développer un modèle de Deep Learning capable de détecter automatiquement la tuberculose à partir d’images de radiographies pulmonaires.

## Jeu de données
Images de tuberculose : 700
Images normales : 3500

## Phase 1 complétée
- Nettoyage des données  
- Analyse exploratoire des données (EDA)  
- Visualisation des radiographies pulmonaires  
- Analyse de la structure du dataset  
- Identification du déséquilibre des classes  

##  Phase 2 complétée
- Chargement des images depuis Google Drive  
- Redimensionnement des images à 224×224 pixels  
- Normalisation des pixels entre 0 et 1  
- Création des variables X et y  
- Division en train/test (80/20)  
- Gestion du déséquilibre avec class weights 

## Prochaine étape
- Entraînement d’un modèle CNN  
- Évaluation des performances  
- Optimisation du modèle  
- Amélioration avec Transfer Learning (MobileNetV2 / ResNet50)

## Outils utilisés
- Python  
- Google Colab  
- TensorFlow / Keras  
- OpenCV  
- Scikit-learn  
