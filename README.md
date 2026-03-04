# Infinix-ClawAnalytics

Lightweight observability layer for OpenClaw-style agents.

## What it solves

Running AI agents without visibility makes it hard to debug latency, tool usage, and cost. This project provides a simple wrapper that captures runtime metrics and writes structured logs, plus a tiny dashboard to visualize basic aggregates.

## Features

- Agent wrapper that measures total latency
- Structured JSONL logs with estimated tokens and cost
- Basic error capture
- Simple dashboard for latency and token usage

## Commercial Analyzer (Enterprise Demo)

This project now includes a modular analyzer for commercial behavior and conversion probability. It generates a synthetic dataset (50k-100k rows), builds customer features, trains a predictive model, clusters behavior, and exposes a dashboard and API for scoring.

### Highlights

- Synthetic data with seasonality, outliers, recurring and inactive customers
- Dynamic risk scoring and churn label prediction
- Behavior clustering and executive performance insights
- Executive dashboard with filters and alerts
- FastAPI scoring endpoint
- CSV wizard with templates, diagnostics, and auto-mapping
- Configurable dashboard labels for any domain
- Calibrated probabilities and cross-validation metrics

### Quickstart

Generate synthetic data:

```
python -m infinix_clawanalytics.enterprise --generate --rows 100000
```

Run the pipeline (feature engineering + model + clusters):

```
python -m infinix_clawanalytics.enterprise --run-pipeline
```

Launch the dashboard:

```
streamlit run infinix_clawanalytics/analyzer/dashboard_app.py
```

### Dashboard modes

- `csv_wizard`: upload CSV, map columns, and run pipeline in-memory.
- `configured`: read `config.yaml` (CSV/DB/API) and run pipeline.
- `scored`: load a previously generated scored file.

The wizard can optionally save the scored output for later use.

### CSV wizard tips

- Use the mapping dropdowns to align your CSV fields to required columns.
- Save templates for reuse across files with the same schema.
- Use the CSV diagnostics to spot invalid dates or flags and download bad rows.
- Use Reset and Delete scored when you want a clean start.

### Configure sources in UI

In `configured` mode, the dashboard can generate `config.yaml` for CSV, Postgres,
MySQL, or API sources from the sidebar form.

Launch the API:

```
uvicorn infinix_clawanalytics.analyzer.api:app --reload
```

### Conecta tu base de datos en 3 pasos

1) Configura la fuente en `config.yaml`:

```yaml
source:
  type: postgres
  table: clientes
  db:
    host: localhost
    port: 5432
    user: user
    password: password
    database: ventas
columns:
  id_cliente: customer_id
  fecha_interaccion: interaction_date
  monto_compra: amount
  conversion: converted
  abandono: churned
  tiempo_respuesta: response_time
  intentos_contacto: attempts
  region: region
  canal: channel
  ejecutivo: executive
```

2) Ejecuta el pipeline con la fuente configurada:

```
python -m infinix_clawanalytics.enterprise --run-pipeline --source postgres
```

3) Abre el dashboard:

```
streamlit run infinix_clawanalytics/analyzer/dashboard_app.py
```

Si es CSV, puedes sobreescribir el archivo asi:

```
python -m infinix_clawanalytics.enterprise --run-pipeline --source csv --file data/mi_archivo.csv
```

### Scored output

By default, the pipeline writes to `artifacts/clientes_scored.parquet`.
The dashboard `scored` mode reads that file.

## Project structure

```
infinix_clawanalytics/
  __init__.py
  monitor.py
  metrics.py
  logger.py
  dashboard.py
  main.py
requirements.txt
README.md
```

## Install

```
pip install -r requirements.txt
```

## Usage

### Wrap an agent

```python
from infinix_clawanalytics import JsonLogger, monitor_agent


def my_agent(prompt: str) -> str:
    return f"Answer: {prompt}"

logger = JsonLogger("logs/agent_logs.jsonl")
wrapped_agent = monitor_agent(my_agent, agent_name="my_agent", logger=logger)

wrapped_agent("hello")
```

### Run demo

```
python -m infinix_clawanalytics.main
```

### Show dashboard

```
python -m infinix_clawanalytics.main --dashboard
```

## Log format

```json
{
  "agent": "agent_name",
  "latency_ms": 123,
  "tool_calls_count": 0,
  "tokens_estimated": 456,
  "cost_estimated": 0.0023,
  "status": "success"
}
```

## Example integration

```python
from infinix_clawanalytics import JsonLogger, monitor_agent

class MyAgent:
    def __call__(self, query: str) -> str:
        return f"Result: {query}"

agent = MyAgent()
logger = JsonLogger("logs/agent_logs.jsonl")
wrapped = monitor_agent(agent, agent_name="openclaw_agent", logger=logger)

wrapped("consulta")
```

## Notes

- Token and cost estimation are placeholders and meant to be replaced.
- The wrapper is generic and does not depend on OpenClaw internals.
