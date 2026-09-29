# Atalhos de desenvolvimento. Cada alvo e um comando Python puro, para que quem
# nao tem make no Windows possa copiar a linha correspondente.
.PHONY: all install lint format typecheck test check lock freeze-reference retrain retrain-check lock-docker freeze-docker api app docker-api docker-app docker-train pipeline-docker shell-docker compose-up compose-down drift-baseline drift-simulated load-test

all:
	python scripts/run_pipeline.py

install:
	pip install -r requirements-train.lock -r requirements-dev.lock
	pip install -e . --no-deps

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

typecheck:
	mypy

test:
	pytest --cov --cov-report=term-missing

check: lint typecheck test

lock:
	pip-compile --generate-hashes --strip-extras -o requirements-api.lock requirements-api.in
	pip-compile --generate-hashes --strip-extras -o requirements-train.lock requirements-train.in
	pip-compile --generate-hashes --strip-extras --constraint requirements-train.lock -o requirements-dev.lock requirements-dev.in
	pip-compile --generate-hashes --strip-extras -o requirements-app.lock requirements-app.in

# Mesmo que "lock", mas resolvido dentro da imagem de treino (Linux, Python 3.11), que e
# o ambiente do CI e do caminho A do README. Gera locks completos, com dependencias
# transitivas e hashes, a partir dos .in. Rodar uma vez e commitar os quatro .lock.
# Congela a referencia de fidelidade da etapa 2 dentro da imagem de treino, que e o
# ambiente fixado. Necessario sempre que pandas, numpy ou scikit-learn mudarem de versao:
# o KNNImputer do codigo original muda a escolha de vizinhos entre releases.
# Etapa 6: avalia o drift da janela contra a referencia do treino, grava a flag em
# evidencias/retraining_flag.json e retreina apenas se ela recomendar. Use retrain-check
# para so decidir e relatar, que e o que um job agendado ou o CI executam.
retrain:
	python scripts/retrain.py

retrain-check:
	python scripts/retrain.py --check-only

freeze-reference:
	docker compose -f docker/docker-compose.yml --profile tools run --rm train \
		python scripts/freeze_notebook_reference.py

lock-docker:
	docker compose -f docker/docker-compose.yml --profile tools run --rm train make lock

# Fotografia do ambiente de treino que executou o pipeline (evidencia de versoes).
freeze-docker:
	docker compose -f docker/docker-compose.yml --profile tools run --rm train sh -c "pip freeze > evidencias/requirements_frozen_train.txt"

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-api:
	docker build -f docker/Dockerfile.api -t hypertension-api:latest .

app:
	streamlit run src/app/main.py

docker-app:
	docker build -f docker/Dockerfile.app -t hypertension-app:latest .

docker-train:
	docker compose -f docker/docker-compose.yml --profile tools build train

pipeline-docker:
	docker compose -f docker/docker-compose.yml --profile tools run --rm train

shell-docker:
	docker compose -f docker/docker-compose.yml --profile tools run --rm train bash

compose-up:
	docker compose -f docker/docker-compose.yml up --build -d

compose-down:
	docker compose -f docker/docker-compose.yml down

drift-baseline:
	python scripts/baseline_scenario.py

drift-simulated:
	python scripts/simulate_drift.py

load-test:
	python scripts/load_test.py --url http://localhost:8000
