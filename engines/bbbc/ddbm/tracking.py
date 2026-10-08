"""Optional experiment tracking for local and cluster runs."""

try:
    import wandb as wandb
except ImportError:

    class _NoOpWandb:
        """Keep correctness independent from the optional W&B UI package."""

        @staticmethod
        def init(*args, **kwargs):
            return None

        @staticmethod
        def watch(*args, **kwargs):
            return None

        @staticmethod
        def log(*args, **kwargs):
            return None

    wandb = _NoOpWandb()
