"""CLI: re-embed stored voice samples from their retained audio using the
current pipeline.

Normally this happens automatically on server startup when the embedding
pipeline changes (storage.maybe_rebuild_on_pipeline_change). This module is the
manual fallback — run it after changing the embedding pipeline if you don't want
to wait for the next server start:

    python -m voice_id.rebuild

Samples enrolled before audio retention existed have no stored audio and are
skipped; those require a manual re-enroll at the mic.
"""

import logging

from .storage import rebuild_embeddings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = rebuild_embeddings()
    print(f"Re-embedded {n} sample(s) from retained audio.")


if __name__ == "__main__":
    main()
