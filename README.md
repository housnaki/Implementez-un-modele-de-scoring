# Projet  7 DS Openclassrooms
## Implémentez un modèle de scoring
### Par: Housna KOUIDRI 
- Lien du Dashboard:https://housnadashboardp7.herokuapp.com/?fbclid=IwAR28DSi57JUYtf5ChTW_8Bifc8aXWCSuiQ0ao7Ar6Owz9nr8Xc56IWRwrLw

## Objectif:
- Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
- Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client

## Données:
- Les données utilisées se trouve à l'adresse suivante: https://www.kaggle.com/c/home-credit-default-risk/data
 
## note méthodologique:
- Pour cette mission nous sommes incités à sélectionner un kernel Kaggle pour nous faciliter la préparation des données nécessaires à l’élaboration du modèle de scoring. ensuite analyser ce kernel et l’adapter pour nous assurer qu’il répond aux besoins de notre mission.

- Le Kernel que j'ai utilisé et adapté ce trouve sur cette adresse: https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features

### Feature engineering:
- Comme mentionné précedemment je me suis inspirée d'un Kernel pour la partie feature engineering:
- 1/Télecharger les données
- 2/Concaténer les différents tableaux
- 3/Créer de nouvelles features

### Imbalanced data:
- On remarque que la donnée à prédire "Target" est fortement désequilibrée (92% de bon clients"class:0") 

![image](https://user-images.githubusercontent.com/94060093/147422796-0d9d0daa-53e9-4231-a37e-a57beaef89bd.png)

### simulation et comparaison des modèles:
- Comparaison de 4 modèles à un algorithme de base "DummyClassifier"; à ces modèles j'ai appliqué un "class_weight" pour traiter les classes non équilibrées/
- les modèles utilisés sont:
- XGBClassifier(scale_pos_weight=9)
- DecisionTreeClassifier(class_weight='balanced')
- RandomForestClassifier(class_weight='balanced')
- BalancedRandomForestClassifier()
![image](https://user-images.githubusercontent.com/94060093/147422761-f142f142-9f2e-4e7b-bde3-f5e04d85fbfc.png)



