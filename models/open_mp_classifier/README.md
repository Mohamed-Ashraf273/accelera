---
title: Open Mp Classifier
emoji: 🐨
colorFrom: pink
colorTo: indigo
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Render

Use `models/open_mp_classifier` as the service root.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Health check path:

```text
/health
```
