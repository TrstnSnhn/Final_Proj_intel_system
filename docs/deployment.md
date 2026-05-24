# PlantGuard Deployment Strategy

PlantGuard is not deployed yet. This document records the recommended deployment path, model artifact handoff plan, runtime strategy, and remaining blockers before a public demo is approved.

## Current Deployment Readiness

- App entrypoint: `web.app:app`
- Local command: `python -m flask --app web.app run`
- Current web routes: `GET /`, `GET /healthz`, `POST /predict`, and Flask static assets
- Current model architecture for the demo: `simple_cnn`
- Current local checkpoint: `experiments/checkpoints/plantvillage_baseline_simple_cnn_best.pt`
- Current local class map: `experiments/checkpoints/plantvillage_baseline_simple_cnn_classes.json`
- Optional artifact path overrides:
  - `PLANTGUARD_CHECKPOINT_PATH`
  - `PLANTGUARD_CLASS_MAP_PATH`
- Current upload rules:
  - Allowed extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
  - Maximum upload size: 5 MB
  - Temporary upload file is deleted after prediction
  - No permanent upload storage
- Production-style server command: `gunicorn --bind 0.0.0.0:${PORT:-7860} web.app:app`
- Docker scaffolding: `Dockerfile` and `.dockerignore`
- Experimental Vercel wrapper entrypoint: `app/app.py` re-exports `web.app:app`

The current checkpoint and class map are ignored by git. They must not be committed to this repository unless that decision is explicitly approved. See `docs/artifacts.md` for artifact validation and handoff details.

## Deployment Options

Pricing and plan limits can change. Check the current provider pricing and resource documentation immediately before deploying.

| Option | Fit | Artifact handling | Complexity | Risk | Portfolio value |
| --- | --- | --- | --- | --- | --- |
| Hugging Face Spaces with Docker | Strong fit for an ML demo because Spaces are built for public model demos and Docker can run Flask plus PyTorch. | Use a separate Hugging Face model repository or a release artifact. Do not commit the checkpoint to this repo. Configure paths with environment variables. | Moderate: needs Dockerfile, Space metadata, production command, and artifact download or mount plan. | Main risks are cold starts, resource limits, and artifact download failure. | High because the demo lives in an ML-focused portfolio surface. |
| Render Flask web service | Good fit for a conventional Flask service. Render Flask docs use Gunicorn for production serving. | Use GitHub Releases, object storage, or a separate model host. Set artifact paths through environment variables. | Low to moderate: Gunicorn start command and artifact strategy are needed. | Main risks are PyTorch dependency size, cold starts, and whether the free or low-cost tier is enough. | Good general web portfolio value. |
| Railway Flask service | Good fit for a simple Flask service with GitHub-based deploys. Railway Flask docs also point to Gunicorn for production serving. | Same external artifact options as Render. | Low to moderate: service setup is straightforward, but artifact hosting still needs a decision. | Main risks are dependency size, usage limits, and startup behavior. | Good general web portfolio value. |
| Fly.io Docker deployment | Technically strong for containerized Flask/PyTorch apps and gives more runtime control. | Use Docker image plus external artifact fetch, mounted volume, or release asset. | Higher: Docker, `fly.toml`, health checks, and platform-specific deploy workflow. | Main risks are extra operational complexity and image size. | Good if the goal is to show container deployment skills. |
| Local-only portfolio demo with GitHub README/screenshots | Already working and lowest risk. | Artifacts stay local and ignored. Users recreate them from docs. | Low: no hosting work. | No public interactive demo. | Good short-term portfolio evidence, but weaker than a live demo. |

Official docs referenced while preparing this plan:

- Hugging Face Spaces overview: https://huggingface.co/docs/hub/spaces-overview
- Hugging Face Docker Spaces: https://huggingface.co/docs/hub/spaces-sdks-docker
- Render Flask deploy guide: https://render.com/docs/deploy-flask
- Railway Flask deploy guide: https://docs.railway.com/guides/flask
- Fly.io app configuration: https://www.fly.io/docs/reference/configuration/

## Recommendation

Primary path: Hugging Face Spaces with Docker.

Reasons:

- It is the best match for an ML portfolio demo.
- It can run Flask with PyTorch without adding a separate frontend build.
- It provides a public demo surface focused on models and AI projects.
- It supports environment variables and secrets through platform settings.
- Docker gives PlantGuard a reproducible Python 3.11 runtime.

Fallback path: keep the local-only portfolio demo with screenshots until artifact hosting is approved. If a conventional web host is preferred later, Render is the simplest Flask fallback because its documented Flask path uses a familiar `pip install -r requirements.txt` plus Gunicorn start command.

Do not deploy until the artifact hosting method and public checkpoint policy are approved.

## Experimental Vercel Note

The earlier `pyproject.toml` Vercel entrypoint attempt was removed because Vercel's Python build can invoke `uv lock`, and a `pyproject.toml` without a `[project]` table is treated as an incomplete Python project. PlantGuard now uses `app/app.py` as a thin Vercel-compatible wrapper around the real Flask app at `web.app:app`.

This remains an experimental compatibility path. Hugging Face Spaces with Docker remains the preferred path for a full ML demo because PlantGuard depends on PyTorch and external model artifacts. A Vercel deployment may still fail later because of serverless limits, package size, cold starts, static asset behavior, or missing checkpoint/class-map artifacts. Do not commit the checkpoint or class map for Vercel; artifact handoff still needs an approved strategy.

## Model Artifact Strategy

Current local artifacts:

```text
experiments/checkpoints/plantvillage_baseline_simple_cnn_best.pt
experiments/checkpoints/plantvillage_baseline_simple_cnn_classes.json
```

The current local files are small enough to handle as explicit demo artifacts, but they still belong outside this repository by default.

The Docker image does not include these artifacts. A future deployment must provide them by mounting files at the default paths, setting `PLANTGUARD_CHECKPOINT_PATH` and `PLANTGUARD_CLASS_MAP_PATH`, or adding an approved startup download step.

Prepared artifact handoff strategy:

1. Keep artifacts ignored locally in `experiments/checkpoints/`.
2. Create a documented artifact handoff process.
3. Prefer a separate Hugging Face model repository for the baseline checkpoint and class map if deploying to Hugging Face Spaces.
4. Use GitHub Releases as the fallback artifact host if Hugging Face model hosting is not approved.
5. Configure deployed artifact locations through environment variables or a small startup download script.
6. Validate artifacts locally with `python src\validate_artifacts.py --expected-classes 15`.
7. Fail clearly if artifacts are missing. Do not fake predictions.

The Flask app already reads:

```text
PLANTGUARD_CHECKPOINT_PATH
PLANTGUARD_CLASS_MAP_PATH
```

If startup download is added later, it should:

- Download only from an approved public artifact URL or authenticated platform secret.
- Write outside the repository working tree when possible, such as `/tmp/plantguard-artifacts` or an app data directory.
- Avoid logging tokens, secret names, credential paths, or signed URLs.
- Verify the files exist before serving prediction requests.

## Runtime and Server Strategy

The Flask development server is for local validation only.

Production-style server:

```bash
gunicorn --bind 0.0.0.0:${PORT:-7860} web.app:app
```

Why Gunicorn:

- It is the standard production WSGI server path documented by common Flask hosts.
- It avoids using Flask's development server in production.
- It keeps the current Flask app structure simple.

Recommended Docker direction for Hugging Face Spaces:

- Base image: Python 3.11 slim image.
- Install `requirements.txt`.
- Use Gunicorn from `requirements.txt`.
- Copy the repo without datasets, checkpoints, logs, caches, or virtual environments.
- Expose the platform port, likely `7860` for Hugging Face Spaces.
- Start with Gunicorn.
- Keep debug mode off.

Do not deploy this image until the model artifact host and runtime path strategy are approved.

## Environment Variables Plan

| Variable | Required | Purpose |
| --- | --- | --- |
| `PLANTGUARD_CHECKPOINT_PATH` | Required in deployed environments unless using the default path inside the image. | Points to the baseline `.pt` checkpoint. |
| `PLANTGUARD_CLASS_MAP_PATH` | Required in deployed environments unless using the default path inside the image. | Points to the baseline class mapping JSON. |
| `PORT` | Platform-dependent. | Port used by Gunicorn or the Docker container. |
| `PLANTGUARD_MAX_UPLOAD_MB` | Future optional. | Would make the 5 MB upload limit configurable. Current code uses a constant. |

Do not create `.env` files in this repository. Configure production variables through the chosen hosting platform.

## Security and UX Checklist

- Keep production debug mode off.
- Keep the 5 MB upload limit unless there is a clear reason to change it.
- Keep the extension allowlist: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
- Continue verifying uploaded files with Pillow before inference.
- Delete temporary upload files after prediction.
- Do not store uploads permanently.
- Do not log secrets, tokens, credential paths, signed artifact URLs, or raw environment values.
- Keep the missing-artifact error clear and user-facing.
- Keep the baseline educational disclaimer visible.
- Avoid diagnosis language and model-quality overclaiming.
- Use `GET /healthz` for process health checks. It does not load the model or require checkpoint files.
- Add deployment validation that does not require a real user upload.

## Recommended Next Deployment Prep Scope

The next deployment-prep phase should still avoid live deployment unless explicitly approved:

1. Approve an external artifact host for the baseline checkpoint and class map.
2. Decide whether deployed artifacts are mounted, copied into a private deployment environment, or downloaded at startup.
3. Add startup artifact download only after an artifact host is approved.
4. Run a local Docker build and container smoke test if Docker is available.
5. Prepare Hugging Face Spaces metadata only after artifact handling is decided.

Do not deploy, push, upload model artifacts, or commit checkpoints unless explicitly approved.

## Remaining Deployment Blockers

- No external artifact host has been approved.
- No public checkpoint/class-map handoff exists, only local validation and documentation.
- No deployment target has been created.
- Docker build/container smoke validation has not been completed because Docker CLI is not available in the current environment.
- Pricing, resource limits, and storage behavior must be checked immediately before actual deployment.
