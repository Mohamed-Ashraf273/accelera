from urllib.parse import urlparse
from accelera.src.deployment.deployment import configure_deployment

class E2EBase:
    def __init__(self):
        self.config = None
        self.graph = None

<<<<<<< HEAD
<<<<<<< HEAD
    def __call__(self, content, config=None, graph=None):
        return self._run(content, config=config, graph=graph)
=======
    def __call__(self, *args, **kwargs):
        return self._run(*args, **kwargs)
>>>>>>> ee06af6 (integrate e2e)
=======
    def __call__(self, content, config=None, graph=None):
        return self._run(content, config=config, graph=graph)
>>>>>>> 002ecff (add e2e pipleline)

    def _is_google_drive_url(self, value: str) -> bool:
        try:
            parsed = urlparse(value.strip())
        except (AttributeError, ValueError):
            return False

        return parsed.scheme in {"http", "https"} and parsed.netloc.lower() in {
            "drive.google.com",
            "docs.google.com",
        }

    def _save_model(self, model, path):
        import joblib

        joblib.dump(model, path)

    def _deploy(self, model):
        df = getattr(self, "df", None)
        config = getattr(self, "config", None)
        target_col = config.get("target_col") if config else None
        configure_deployment(model, df=df, target_col=target_col)

    def _run(self, content, config=None, graph=None):
        raise NotImplementedError(
            "This data type is not supported for Accelera E2E."
        )
