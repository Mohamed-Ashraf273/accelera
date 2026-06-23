from urllib.parse import urlparse


class E2EBase:
    def __init__(self):
        self.config = None
        self.graph = None

    def __call__(self, content, config=None, graph=None):
        return self._run(content, config=config, graph=graph)

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
        # To be implemented
        pass

    def _run(self, content, config=None, graph=None):
        raise NotImplementedError(
            "This data type is not supported for Accelera E2E."
        )
