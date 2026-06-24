"""CLI: re-embed stored voice samples from their retained audio using the
current pipeline.

Normally this happens automatically on server startup when the embedding
pipeline changes (storage.maybe_rebuild_on_pipeline_change). This module is the
manual fallback — run it after changing the embedding pipeline if you don't want
to wait for the next server start:

    python -m voice_id.rebuild
    python -m voice_id.rebuild --prune    # also drop unrebuildable (no-audio) samples

Samples enrolled before audio retention existed have no stored audio. They can't
be re-embedded, so --prune removes them (they can't be part of a persistent
profile); without --prune they're left in place and skipped by the rebuild.
"""

import argparse
import logging

from .storage import prune_unrebuildable, rebuild_embeddings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Re-embed voice samples from retained audio.")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete samples with no retained audio (unrebuildable legacy enrollments) before rebuilding.",
    )
    args = parser.parse_args()

    if args.prune:
        pruned = prune_unrebuildable()
        print(f"Pruned {pruned} unrebuildable sample(s) (no retained audio).")

    n = rebuild_embeddings()
    print(f"Re-embedded {n} sample(s) from retained audio.")


if __name__ == "__main__":
    main()
