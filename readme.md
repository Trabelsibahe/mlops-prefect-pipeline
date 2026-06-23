examen sur mlflow
cycle de vie : entrainement du model, save model, load model, evaluate model
n iterations sur modeles : experimentations
prob?
experimentation, reproduction, deployer, gestion des versions
architecture
composant
projectflow vs modelflow(sklearn, pytorch, tensorflow...)
diap : 23
# FastAPI + MLflow + Docker + Prefect

Ce projet expose une API FastAPI de prediction de churn et integre MLflow pour le tracking des runs et des artefacts.

## 1) Build Docker

Nom d'image attendu pour l'atelier:

- format: prenom_nom_classe_mlops
- exemple: bahe_trabelsi_4twinf_mlops

Commande de build:

```bash
docker build -t prenom_nom_classe_mlops:latest .
```

Lister les images:

```bash
docker images
```

## 2) Test local du conteneur

### Option A: modele local dans le projet

Si le fichier classifier.joblib existe dans le projet:

```bash
docker run --rm -p 8000:8000 prenom_nom_classe_mlops:latest
```

### Option B: artefact MLflow dans le conteneur

L'application peut charger le modele directement depuis MLflow avec:

- MLFLOW_MODEL_URI (ex: runs:/<RUN_ID>/saved_model/classifier.joblib)
- ou MLFLOW_RUN_ID + MLFLOW_ARTIFACT_PATH

Exemple:

```bash
docker run --rm -p 8000:8000 \
	-e MLFLOW_TRACKING_URI="sqlite:///mlflow.db" \
	-e MLFLOW_RUN_ID="<RUN_ID>" \
	-e MLFLOW_ARTIFACT_PATH="saved_model/classifier.joblib" \
	prenom_nom_classe_mlops:latest
```

Tester l'API:

- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## 3) Push Docker Hub

```bash
docker login
docker tag prenom_nom_classe_mlops:latest <dockerhub_user>/prenom_nom_classe_mlops:latest
docker push <dockerhub_user>/prenom_nom_classe_mlops:latest
```

Verifier ensuite l'image sur Docker Hub.

## 4) Automatisation Prefect (CD)

Le flow CD est ajoute dans pipeline_prefect.py avec les taches:

- docker_build_image
- docker_run_container
- docker_login
- docker_tag_image
- docker_push_image

Variables d'environnement utilisees par le flow CD:

- DOCKER_IMAGE_NAME (defaut: prenom_nom_classe_mlops)
- DOCKER_IMAGE_TAG (defaut: latest)
- DOCKER_CONTAINER_NAME (defaut: fastapi-mlflow-app)
- DOCKER_LOCAL_PORT (defaut: 8000)
- DOCKERHUB_USERNAME
- DOCKERHUB_TOKEN

Execution:

```bash
python pipeline_prefect.py --flow cd
```

Le fichier deploiement_prefect.py inclut aussi le deployment Prefect de ce flow CD.
