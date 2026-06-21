# tb-detection-deep-learning

## Description du projet

Ce projet vise à développer un modèle de Deep Learning capable de détecter automatiquement la tuberculose à partir d’images de radiographies pulmonaires.

---

## Jeu de données

- Images de tuberculose : 700  
- Images normales : 3500  

---

## Phase 1 complétée

- Nettoyage des données  
- Analyse exploratoire des données (EDA)  
- Visualisation des radiographies pulmonaires  
- Analyse de la structure du dataset  
- Identification du déséquilibre des classes  

---

## Phase 2 complétée

- Chargement des images depuis Google Drive  
- Redimensionnement des images à 224×224 pixels  
- Normalisation des pixels entre 0 et 1  
- Création des variables X et y  
- Division en train/test (80/20)  
- Gestion du déséquilibre avec class weights  

---

## Phase 3 complétée – Modélisation

Deux approches ont été développées :

### CNN classique
- Architecture convolutionnelle simple  
- Accuracy : **87%**

### Transfer Learning (MobileNetV2)
- Accuracy : **94%**
- Meilleure généralisation et performance

---

## Évaluation du modèle

Le modèle final a été évalué à l’aide de :

- Matrice de confusion  
- Courbe ROC (AUC)  
- Classification report (precision, recall, F1-score)

Les résultats montrent une très bonne capacité de détection de la tuberculose avec un recall élevé, ce qui est essentiel dans un contexte médical.

---

## Prochaine étape

- Optimisation et fine-tuning du modèle  
- Amélioration des performances  
- Déploiement (API ou interface web)  
- Amélioration possible avec ResNet50  

---

## Outils utilisés

- Python  
- Google Colab  
- TensorFlow / Keras  
- OpenCV  
- Scikit-learn  
- Matplotlib / Seaborn  
