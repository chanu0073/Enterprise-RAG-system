PY ?= .venv/bin/python
API_PORT ?= 8000
UI_PORT ?= 8501

.PHONY: install dev-backend dev-frontend test up up-gpu down logs

install:
	$(PY) -m pip install -r requirements-dev.txt -r frontend/requirements.txt

dev-backend:
	cd backend && ../$(PY) -m uvicorn app.main:app --reload --port $(API_PORT)

dev-frontend:
	cd frontend && EKIP_API_URL=http://localhost:$(API_PORT) ../$(PY) -m streamlit run app.py --server.port $(UI_PORT)

test:
	$(PY) -m pytest

up:
	docker compose up -d --build

up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

down:
	docker compose down

logs:
	docker compose logs -f backend frontend
