#!/usr/bin/env python3
"""
Download the transcribed ASR subset of facebook/voxpopuli and convert it into
audio files + JSONL manifests.

Configs are language codes (cs de en es et fi fr hr hu it lt nl pl ro sk sl, plus
en_accented = test-only accented English). The `multilang` config is skipped: it is
the union of the per-language configs. Output:
    OUT/audio/<lang>/<split>/<session>/<audio_id>.wav
    OUT/manifests/<lang>/<split>.jsonl

<session> is the plenary-session prefix of audio_id (the part before the first "_").

JSONL fields: audio_filepath, sampling_rate, split, audio_id, session, language (ISO code),
raw_text, normalized_text, gender, speaker_id, is_gold_transcript, accent

Not covered: the unlabelled 400k-hour audio and the speech-to-speech interpretation
data, which are distributed via https://github.com/facebookresearch/voxpopuli.

Requirements: pip install huggingface_hub pyarrow   (optionally: hf_transfer)
"""

import argparse
import sys

import utils

# ClassLabel order of the `language` column, as declared in the dataset card.
LANG_NAMES = ["en", "de", "fr", "es", "pl", "it", "ro", "hu", "cs", "nl",
              "fi", "hr", "sk", "sl", "et", "lt", "en_accented"]


def row_fn(row: dict, config: str, split: str):
    audio_id = row["audio_id"]
    session = audio_id.split("_", 1)[0] if "_" in audio_id else "unknown"
    lang = row.get("language")
    if isinstance(lang, int):
        lang = LANG_NAMES[lang] if 0 <= lang < len(LANG_NAMES) else config
    return session, audio_id, {"session": session, "language": lang or config}


SPEC = utils.DatasetSpec(repo_id="facebook/voxpopuli", row_fn=row_fn,
                         exclude_configs=frozenset({"multilang"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    utils.add_common_args(ap, default_out="VoxPopuli")
    ap.add_argument("--languages", nargs="*", default=None,
                    help="subset of language configs: " + " ".join(LANG_NAMES))
    args = ap.parse_args()

    shards = utils.list_shards(SPEC.repo_id, args.revision, SPEC.exclude_configs)
    if args.list:
        utils.print_listing(shards)
        return

    available = {c for c, _ in shards}
    configs = set(available)
    if args.languages:
        unknown = set(args.languages) - available
        if unknown:
            sys.exit(f"unknown languages: {sorted(unknown)}; available: {sorted(available)}")
        configs &= set(args.languages)

    utils.run(SPEC, args, shards, configs)


if __name__ == "__main__":
    main()
