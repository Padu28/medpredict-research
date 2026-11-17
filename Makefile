.PHONY: install dev pipeline api dashboard test lint docker clean

install:        ## Install runtime dependencies (editable)
	pip install -r requirements.txt && pip install -e .

dev:            ## Install dev tooling
	pip install -r requirements-dev.txt && pip install -e .

pipeline:       ## Full research pipeline
	python -m medpredict.cli pipeline

api:            ## FastAPI on :8003 (docs at /docs)
	uvicorn api.main:app --reload --port 8003

dashboard:      ## Streamlit dashboard on :8504
	streamlit run dashboard/app.py --server.port 8504

test:
	pytest --cov=medpredict --cov-report=term-missing

lint:
	ruff check src api dashboard tests && ruff format --check src api dashboard tests

docker:
	docker build -f docker/Dockerfile -t medpredict-research:latest . && docker run -p 8003:8003 -p 8504:8504 medpredict-research:latest

clean:
	rm -rf data/processed/*.db models/*.joblib reports/figures/*.png .pytest_cache .ruff_cache
