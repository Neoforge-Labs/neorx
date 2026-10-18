# REST API

NeoRx ships a FastAPI service wrapping the pipeline, so a run can be
started and polled over HTTP rather than held in a Python process.

```bash
neorx serve --port 8000
```

Interactive docs at `http://localhost:8000/docs`, the OpenAPI schema at
`/openapi.json`, and ReDoc at `/redoc`.

## Endpoints

| Method | Path | Does |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/run` | Start a full pipeline run; returns a job ID |
| `GET` | `/status/{job_id}` | Progress and status of a run |
| `GET` | `/report/{job_id}` | The HTML report for a finished run |
| `POST` | `/graph` | Build the disease graph only |
| `POST` | `/identify` | Identify causal targets only |
| `POST` | `/screen-target` | Screen candidates against one target |

`/run` is asynchronous: it returns a job ID immediately and the work
proceeds in the background. `/graph`, `/identify` and `/screen-target`
are synchronous and block until done — which for `/graph` means minutes
against eight live services.

```python
>>> from fastapi.testclient import TestClient
>>> from neorx.core.api import app
>>> client = TestClient(app)
>>> response = client.get("/health")
>>> response.status_code, response.json()
(200, {'status': 'ok', 'service': 'NeoRx'})

```

A missing job is a 404 with a message, not an empty success:

```python
>>> response = client.get("/status/not-a-real-job")
>>> response.status_code
404
>>> response.json()
{'detail': 'Job not-a-real-job not found.'}

```

## Request models

Every endpoint takes a Pydantic model, so a malformed request is
rejected with a 422 and a field-level explanation rather than failing
somewhere inside the pipeline.

```python
>>> from neorx import (RunRequest, GraphRequest, IdentifyRequest,
...                    ScreenTargetRequest, StatusResponse)
>>> list(RunRequest.model_fields)
['disease', 'top_n_targets', 'candidates_per_target', 'allow_mocks']
>>> list(IdentifyRequest.model_fields)
['disease', 'top_n', 'allow_mocks']
>>> list(ScreenTargetRequest.model_fields)
['target_uniprot_id', 'target_pdb_id', 'n_candidates', 'causal_confidence']
>>> list(StatusResponse.model_fields)
['job_id', 'status', 'progress_pct', 'current_step', 'error']

```

The bounds are enforced, not advisory:

```python
>>> request = RunRequest(disease="HIV")
>>> request.top_n_targets, request.candidates_per_target, request.allow_mocks
(5, 1000, False)
>>> import pydantic
>>> try:
...     RunRequest(disease="HIV", top_n_targets=500)
... except pydantic.ValidationError as exc:
...     print("rejected:", "less_than_equal" in str(exc))
rejected: True
>>> try:
...     RunRequest(disease="")
... except pydantic.ValidationError as exc:
...     print("rejected:", "at least 1 character" in str(exc))
rejected: True

```

`top_n_targets` is capped at 50 and `candidates_per_target` at 50,000 —
limits on a public endpoint rather than suggestions. Note that
`RunRequest.candidates_per_target` defaults to **1000**, where the CLI's
`--candidates` defaults to 100.

`allow_mocks` is exposed on `/run`, `/graph` and `/identify` and defaults
to `False`, the same as everywhere else. Enabling it over HTTP means the
response can contain curated hand-written data; see the
[CLI note](cli.md#neorx-run).

## Using it

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"disease": "HIV", "top_n_targets": 5}'

curl http://localhost:8000/status/<job_id>

curl -X POST http://localhost:8000/identify \
  -H "Content-Type: application/json" \
  -d '{"disease": "HIV", "top_n": 10}'
```

## Job state

Jobs are held in the service process. With `DATABASE_URL` set they are
also written to PostgreSQL, which is what the Docker Compose setup
provides — see [Install](install.md#docker). Without it, restarting the
service loses in-flight job state.

!!! warning "There is no authentication"
    The service has no auth, no rate limiting and no request quotas, and
    `neorx serve` binds `0.0.0.0` by default. Every endpoint can start
    minutes of work against third-party APIs on an unauthenticated
    request. Run it behind something, or bind it to localhost.

## Reports

`GET /report/{job_id}` returns the HTML report for a completed job: the
causal knowledge graph visualisation, the target reasoning, and the
ranked candidate table. The same report is written to disk by
`run_pipeline(..., generate_report=True)` and its path appears on
`PipelineResult.report_path`.
