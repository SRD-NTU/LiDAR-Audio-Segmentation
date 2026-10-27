"""Raw four-channel audio + LiDAR inference for ActiveView V1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "code"))
for key, folder in [("XDG_CACHE_HOME", "xdg"), ("TORCH_HOME", "torch"),
                    ("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba")]:
    os.environ.setdefault(key, str(HERE / "outputs/.cache" / folder))
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", USE_TF="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")

import numpy as np
import soundfile as sf
import torch

from acoustic_wm.audio.predictive_features import covariance48_from_wave
from acoustic_wm.deployment_menu import load_menu, load_observation, check_source_consistency
from acoustic_wm.frontend import RawObservationEncoder
from acoustic_wm.v1_pipeline import ActiveViewV1


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(menu_path, output_path, *, checkpoint=None, device="cuda", max_queries=1,
        max_translation_m=3.0, grouping_threshold=0.9, imagined_outcomes=8,
        allow_v0_bootstrap=False, encoder=None, model=None):
    torch.set_num_threads(1)
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = False
    menu, _, valid, costs = load_menu(menu_path)
    xyz, wave, rate = load_observation(menu["reference"])
    config = json.loads((HERE / "config.json").read_text())
    config["paths"]["dataset"] = str(HERE / "assets/sensor")
    if encoder is None:
        encoder = RawObservationEncoder(config, device)
    row = encoder.encode_arrays(xyz, wave, rate)
    observed, reliability = covariance48_from_wave(wave)
    if model is None:
        model = ActiveViewV1(HERE, device, checkpoint=checkpoint,
                             allow_v0_bootstrap=allow_v0_bootstrap,
                             grouping_threshold=grouping_threshold,
                             imagined_outcomes=imagined_outcomes)

    def acquire(query):
        selected = menu["observations"][query]
        check_source_consistency(menu["reference"], selected)
        # ActiveView never opens the second LiDAR path.
        second, second_rate = sf.read(selected["audio"], dtype="float32", always_2d=True)
        if second_rate != 44100 or second.shape[1] != 4 or not np.isfinite(second).all():
            raise ValueError("Selected future audio must be finite 44.1 kHz, four-channel")
        return covariance48_from_wave(second)

    result = model.run(row, observed, reliability,
                       menu["reference"]["receiver_to_world"],
                       [x["receiver_to_world"] for x in menu["observations"]],
                       acquire, valid=valid, costs=costs, max_queries=max_queries,
                       max_translation_m=max_translation_m)
    result["observation_menu_json"] = np.str_(json.dumps(menu, sort_keys=True))
    result["checkpoint_sha256"] = np.str_(sha(checkpoint) if checkpoint is not None else "v0-bootstrap-untrained")
    result["selected_observation_ids"] = np.asarray(
        [str(menu["observations"][i]["id"]) for i in result["chosen_actions"]])
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **result)
    return {"output": str(destination), "checkpoint_identity": model.training_identity,
            "chosen_actions": result["chosen_actions"].tolist(),
            "selected_object": int(result["selected_object"]),
            "selected_group": int(result["selected_group"]),
            "foreground_points": int(result["sound_mask"].sum()),
            "audio_observations": int(result["audio_observations"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--menu", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", help="Trained V1 checkpoint; required unless bootstrapping")
    parser.add_argument("--allow-v0-bootstrap", action="store_true",
                        help="Functional check only: V1 verifier has not been trained")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-queries", type=int, default=1)
    parser.add_argument("--max-translation-m", type=float, default=3.0)
    parser.add_argument("--grouping-threshold", type=float, default=0.9)
    parser.add_argument("--imagined-outcomes", type=int, default=8)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    print(json.dumps(run(args.menu, args.output, checkpoint=checkpoint, device=args.device,
                         max_queries=args.max_queries, max_translation_m=args.max_translation_m,
                         grouping_threshold=args.grouping_threshold,
                         imagined_outcomes=args.imagined_outcomes,
                         allow_v0_bootstrap=args.allow_v0_bootstrap)))


if __name__ == "__main__":
    main()
