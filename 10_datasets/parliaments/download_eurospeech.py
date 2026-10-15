#!/usr/bin/env python3
"""
Download disco-eth/EuroSpeech and convert it into audio files + JSONL manifests.

22 configs, one per country/parliament, each monolingual. Output:
    OUT/audio/<country>/<split>/<video_id>/<key>.wav
    OUT/manifests/<country>/<split>.jsonl

JSONL fields: audio_filepath, sampling_rate, split, key, country, language, video_id,
transcript_id, start_seconds, end_seconds, duration_seconds, asr_transcript,
human_transcript, cer, wer, original_transcript_start_idx, original_transcript_end_idx

WARNING: the full corpus is several TB (denmark, croatia ~600 GB each).
Use --countries / --languages / --splits / --max-shards to fetch a subset first.

Requirements: pip install huggingface_hub pyarrow   (optionally: hf_transfer)
"""

import argparse
import sys

import utils

COUNTRY_LANG = {
    "bosnia-herzegovina": "bs", "bulgaria": "bg", "croatia": "hr", "denmark": "da", "estonia": "et",
    "finland": "fi", "france": "fr", "germany": "de", "greece": "el", "iceland": "is", "italy": "it",
    "latvia": "lv", "lithuania": "lt", "malta": "mt", "norway": "no", "portugal": "pt", "serbia": "sr",
    "slovakia": "sk", "slovenia": "sl", "sweden": "sv", "uk": "en", "ukraine": "uk",
}


def row_fn(row: dict, config: str, split: str):
    return row.get("video_id"), row["key"], {}


SPEC = utils.DatasetSpec(repo_id="disco-eth/EuroSpeech", row_fn=row_fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    utils.add_common_args(ap, default_out="EuroSpeech")
    ap.add_argument("--countries", nargs="*", default=None, help="subset of configs, e.g. germany uk slovakia")
    ap.add_argument("--languages", nargs="*", default=None,
                    help="select by ISO code: " + " ".join(sorted(set(COUNTRY_LANG.values()))))
    args = ap.parse_args()

    shards = utils.list_shards(SPEC.repo_id, args.revision, SPEC.exclude_configs)
    if args.list:
        utils.print_listing(shards, lambda c: COUNTRY_LANG.get(c, "?"))
        return

    available = {c for c, _ in shards}
    configs = set(available)
    if args.countries:
        unknown = set(args.countries) - available
        if unknown:
            sys.exit(f"unknown countries: {sorted(unknown)}; available: {sorted(available)}")
        configs &= set(args.countries)
    if args.languages:
        unknown = set(args.languages) - set(COUNTRY_LANG.values())
        if unknown:
            sys.exit(f"unknown languages: {sorted(unknown)}; available: {sorted(set(COUNTRY_LANG.values()))}")
        configs &= {c for c, l in COUNTRY_LANG.items() if l in args.languages}

    utils.run(SPEC, args, shards, configs)


if __name__ == "__main__":
    main()
