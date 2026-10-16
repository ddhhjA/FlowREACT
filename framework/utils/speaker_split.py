"""Audited identity and recording manifests for isolated FlowREACT experiments."""

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath


IDENTITY_FIELDS = ("source_split", "recording_id", "speaker_id", "listener_id")
PAIR_FIELDS = (
    "source_split", "recording_id", "speaker_id", "listener_id",
    "speaker_path", "listener_path",
)
PARTITIONS = ("train", "val", "seen", "unseen")
RECORDING_FIELDS = ("source_split", "recording_id", "speaker_path", "listener_path")
RECORDING_PARTITIONS = ("train", "val", "heldout")


def protocol_partitions(protocol):
    if protocol == "speaker_independent":
        return PARTITIONS
    if protocol == "recording_disjoint":
        return RECORDING_PARTITIONS
    raise ValueError("Unknown split protocol: {}".format(protocol))


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path, fields):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(fields) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Missing CSV columns in {}: {}".format(path, sorted(missing)))
        rows = [{key: (row.get(key) or "").strip() for key in fields} for row in reader]
    if not rows:
        raise ValueError("Empty manifest: {}".format(path))
    for number, row in enumerate(rows, 2):
        if not all(row.values()):
            hint = "use real participant IDs" if "speaker_id" in fields else "complete the recording/path fields"
            raise ValueError("Unfilled required field at {}:{}; {}".format(path, number, hint))
    return rows


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discover_recordings(root_dir, source_splits=("train", "val")):
    root = Path(root_dir)
    recordings = {}
    for split in source_splits:
        if split not in {"train", "val", "test"}:
            raise ValueError("Invalid physical source split: {}".format(split))
        speaker_dir = root / split / "coefficients" / "speaker"
        for path in sorted(speaker_dir.glob("session*/*.npy")):
            relative = path.relative_to(root / split / "coefficients").with_suffix("")
            listener = PurePosixPath("listener", relative.parts[1], relative.name)
            if not (root / split / "coefficients" / listener).with_suffix(".npy").is_file():
                raise FileNotFoundError("Missing paired listener coefficients: {}".format(listener))
            key = (split, path.stem)
            recording = recordings.setdefault(key, {
                "source_split": split, "recording_id": path.stem, "sessions": [],
            })
            recording["sessions"].append((relative.as_posix(), listener.as_posix()))
    if not recordings:
        raise ValueError("No paired coefficients found under {}/<split>/coefficients/speaker/session*".format(root))
    return recordings


def identity_template(root_dir, output, source_splits=("train", "val")):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Will not overwrite an identity mapping: {}".format(output))
    recordings = discover_recordings(root_dir, source_splits)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, IDENTITY_FIELDS, [
        {"source_split": split, "recording_id": recording_id,
         "speaker_id": "", "listener_id": ""}
        for split, recording_id in sorted(recordings)
    ])
    return len(recordings)


def _sequence_path(value):
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value
            or len(path.parts) != 3 or path.parts[0] not in {"speaker", "listener"}
            or not path.parts[1].startswith("session") or path.suffix):
        raise ValueError("Expected role/session*/recording path without an extension: {}".format(value))
    return path


def validate_partitions(partitions, unseen_ids):
    if set(partitions) != set(PARTITIONS):
        raise ValueError("Expected train, val, seen and unseen manifests")
    if not unseen_ids:
        raise ValueError("At least one actual held-out speaker is required")
    memberships = {}
    sequence_memberships = {}
    counts = {}
    for partition, rows in partitions.items():
        if not rows:
            raise ValueError("Partition {} is empty; adjust the identity/recording split".format(partition))
        sample_keys = set()
        for row in rows:
            if row["source_split"] not in {"train", "val", "test"}:
                raise ValueError("Invalid physical source_split")
            speaker_path = _sequence_path(row["speaker_path"])
            listener_path = _sequence_path(row["listener_path"])
            if (speaker_path.parts[1:] != listener_path.parts[1:]
                    or speaker_path.parts[0] == listener_path.parts[0]):
                raise ValueError("Speaker and listener must be paired within the same recording/session")
            if speaker_path.name != row["recording_id"]:
                raise ValueError("recording_id must match the physical recording filename")
            if row["speaker_id"] == row["listener_id"]:
                raise ValueError("A dyadic pair must contain two different participant IDs")
            sample_key = (row["source_split"], row["speaker_path"])
            if sample_key in sample_keys:
                raise ValueError("Duplicate input sample in {}: {}".format(partition, sample_key))
            sample_keys.add(sample_key)
            recording = row["recording_id"]
            if recording in memberships and memberships[recording] != partition:
                raise ValueError("Recording leakage across partitions: {}".format(recording))
            memberships[recording] = partition
            for field in ("speaker_path", "listener_path"):
                key = (row["source_split"], row[field])
                if key in sequence_memberships and sequence_memberships[key] != partition:
                    raise ValueError("Sequence leakage across partitions: {}".format(key))
                sequence_memberships[key] = partition
        counts[partition] = {
            "samples": len(rows), "recordings": len({row["recording_id"] for row in rows}),
            "speakers": len({row["speaker_id"] for row in rows}),
        }
    train_ids = {row[field] for row in partitions["train"] for field in ("speaker_id", "listener_id")}
    dev_ids = {row[field] for row in partitions["val"] for field in ("speaker_id", "listener_id")}
    held_out = set(unseen_ids)
    if held_out & (train_ids | dev_ids):
        raise ValueError("Held-out participant occurs in training/development in either role")
    train_speakers = {row["speaker_id"] for row in partitions["train"]}
    if not {row["speaker_id"] for row in partitions["seen"]} <= train_speakers:
        raise ValueError("Seen group contains an input speaker absent from training inputs")
    if not {row["speaker_id"] for row in partitions["unseen"]} <= held_out:
        raise ValueError("Unseen group contains a non-held-out input speaker")
    return counts


def prepare_split(root_dir, identities, output_dir, source_splits=("train", "val"),
                  seed=1234, unseen_fraction=0.2, val_fraction=0.15, seen_fraction=0.15,
                  unseen_speakers=None, verify_files=True):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new split directory; existing experiment manifests are immutable")
    if not (0 < unseen_fraction < 1 and 0 < val_fraction < 1 and 0 < seen_fraction < 1
            and val_fraction + seen_fraction < 1):
        raise ValueError("Invalid held-out-speaker or recording split fractions")
    recordings = discover_recordings(root_dir, source_splits)
    mapping = {}
    participant_assignments = {}
    for row in read_csv(identities, IDENTITY_FIELDS):
        key = (row["source_split"], row["recording_id"])
        if key in mapping:
            raise ValueError("Duplicate identity row: {}".format(key))
        mapping[key] = row
        if row["speaker_id"] == row["listener_id"]:
            raise ValueError("Speaker and listener IDs cannot be identical")
        assignment = (row["speaker_id"], row["listener_id"])
        previous = participant_assignments.setdefault(row["recording_id"], assignment)
        if previous != assignment:
            raise ValueError("Inconsistent identities for recording {}".format(row["recording_id"]))
    missing = set(recordings) - set(mapping)
    if missing:
        raise ValueError("Missing identities for {} recordings; example: {}".format(len(missing), sorted(missing)[0]))
    rng = random.Random(seed)
    primary_speakers = sorted({mapping[key]["speaker_id"] for key in recordings})
    if unseen_speakers:
        held_out = set(unseen_speakers)
        unknown = held_out - set(primary_speakers)
        if unknown:
            raise ValueError("Requested IDs are not source speakers: {}".format(sorted(unknown)))
    else:
        if len(primary_speakers) < 2:
            raise ValueError("Need at least two actual source speaker identities")
        held_out = set(rng.sample(primary_speakers, min(len(primary_speakers) - 1,
                                                      max(1, round(len(primary_speakers) * unseen_fraction)))))
    # A recording is indivisible even when the archive contains multiple sessions/directions.
    known_recordings = sorted({key[1] for key in recordings
                               if not held_out & {mapping[key]["speaker_id"], mapping[key]["listener_id"]}})
    if len(known_recordings) < 3:
        raise ValueError("Need at least three non-held-out recordings for train, val and seen")
    by_speaker = defaultdict(set)
    for key in recordings:
        if key[1] in known_recordings:
            by_speaker[mapping[key]["speaker_id"]].add(key[1])
    # Keep at least one complete training recording for every eligible seen speaker.
    anchors = {rng.choice(sorted(names)) for _, names in sorted(by_speaker.items())}
    remaining_recordings = sorted(set(known_recordings) - anchors)
    if len(remaining_recordings) < 2:
        raise ValueError("Need multiple recordings per known speaker to create nonoverlapping seen/development groups")
    rng.shuffle(remaining_recordings)
    val_count = max(1, round(len(known_recordings) * val_fraction))
    seen_count = max(1, round(len(known_recordings) * seen_fraction))
    if val_count + seen_count > len(remaining_recordings):
        raise ValueError("Too few non-anchor recordings; reduce val_fraction/seen_fraction")
    groups = {name: "train" for name in known_recordings}
    groups.update({name: "val" for name in remaining_recordings[:val_count]})
    groups.update({name: "seen" for name in remaining_recordings[val_count:val_count + seen_count]})
    partitions = {name: [] for name in PARTITIONS}
    missing_features = []
    seen_sources = set()
    for key, recording in sorted(recordings.items()):
        identity = mapping[key]
        is_unseen = bool(held_out & {identity["speaker_id"], identity["listener_id"]})
        partition = "unseen" if is_unseen else groups[key[1]]
        for speaker_path, listener_path in recording["sessions"]:
            # Deduplicate archives containing the same recording in more than one physical split.
            physical_key = (key[1], speaker_path)
            if physical_key in seen_sources:
                raise ValueError("Recording/session duplicated across source archives: {}".format(physical_key))
            seen_sources.add(physical_key)
            if verify_files:
                for sequence in (speaker_path, listener_path):
                    for modality in ("audio-features", "facial-attributes", "coefficients"):
                        path = (Path(root_dir) / key[0] / modality / sequence).with_suffix(".npy")
                        if not path.is_file():
                            missing_features.append(str(path))
            for reverse in (False, True):
                row = {"source_split": key[0], "recording_id": key[1],
                       "speaker_id": identity["listener_id" if reverse else "speaker_id"],
                       "listener_id": identity["speaker_id" if reverse else "listener_id"],
                       "speaker_path": listener_path if reverse else speaker_path,
                       "listener_path": speaker_path if reverse else listener_path}
                if partition != "unseen" or row["speaker_id"] in held_out:
                    partitions[partition].append(row)
    if missing_features:
        raise FileNotFoundError("Missing {} required feature files; first: {}".format(len(missing_features), missing_features[0]))
    train_speakers = {row["speaker_id"] for row in partitions["train"]}
    original_seen_count = len(partitions["seen"])
    partitions["seen"] = [row for row in partitions["seen"] if row["speaker_id"] in train_speakers]
    counts = validate_partitions(partitions, held_out)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in PARTITIONS:
        path = output / (name + ".csv")
        write_csv(path, PAIR_FIELDS, partitions[name])
        hashes[name] = file_digest(path)
    split_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
    audit = {
        "protocol": "speaker_independent", "split_id": split_id, "split_seed": seed,
        "manifest_sha256": hashes, "identity_sha256": file_digest(identities),
        "unseen_speaker_ids": sorted(held_out), "counts": counts,
        "seen_directions_removed": original_seen_count - len(partitions["seen"]),
        "source_splits": list(source_splits), "files_verified": verify_files,
        "listener_identity_disjoint": False,
        "scope": "Unseen input speakers; target listener identities may overlap across groups.",
    }
    (output / "split.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def validate_recording_partitions(partitions):
    if set(partitions) != set(RECORDING_PARTITIONS):
        raise ValueError("Expected train, val and heldout recording manifests")
    memberships = {}
    physical_sequences = set()
    counts = {}
    for partition, rows in partitions.items():
        if not rows:
            raise ValueError("Partition {} is empty; adjust the recording fractions".format(partition))
        directed_inputs = set()
        for row in rows:
            if row["source_split"] not in {"train", "val", "test"}:
                raise ValueError("Invalid physical source_split")
            speaker = _sequence_path(row["speaker_path"])
            listener = _sequence_path(row["listener_path"])
            if (speaker.parts[1:] != listener.parts[1:]
                    or speaker.parts[0] == listener.parts[0] or speaker.name != row["recording_id"]):
                raise ValueError("Paths must pair opposite roles in the same recording/session")
            recording = row["recording_id"]
            if recording in memberships and memberships[recording] != partition:
                raise ValueError("Recording leakage across partitions: {}".format(recording))
            memberships[recording] = partition
            key = (row["source_split"], row["speaker_path"])
            if key in directed_inputs:
                raise ValueError("Duplicate input sample: {}".format(key))
            directed_inputs.add(key)
            # A repeated recording/session in two physical archives is ambiguous.
            physical_key = row["speaker_path"]
            if physical_key in physical_sequences:
                raise ValueError("Recording/session duplicated across source archives: {}".format(physical_key))
            physical_sequences.add(physical_key)
        for row in rows:
            if (row["source_split"], row["listener_path"]) not in directed_inputs:
                raise ValueError("Recording evaluation must retain both input directions")
        counts[partition] = {
            "samples": len(rows), "pairs": len(rows) // 2,
            "recordings": len({row["recording_id"] for row in rows}), "speakers": None,
        }
    return counts


def prepare_recording_split(root_dir, output_dir, source_splits=("train", "val"),
                            seed=1234, test_fraction=0.2, val_fraction=0.15, verify_files=True):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new split directory; existing experiment manifests are immutable")
    if len(set(source_splits)) != len(source_splits):
        raise ValueError("Physical source splits must be unique")
    if not (0 < test_fraction < 1 and 0 < val_fraction < 1 and test_fraction + val_fraction < 1):
        raise ValueError("Invalid recording fractions; leave a nonempty training partition")
    recordings = discover_recordings(root_dir, source_splits)
    names = sorted({key[1] for key in recordings})
    if len(names) < 3:
        raise ValueError("Need at least three complete recordings for train, val and heldout")
    rng = random.Random(seed)
    rng.shuffle(names)
    test_count = max(1, round(len(names) * test_fraction))
    val_count = max(1, round(len(names) * val_fraction))
    if test_count + val_count >= len(names):
        raise ValueError("Fractions leave no training recordings; reduce test_fraction/val_fraction")
    groups = {name: "heldout" for name in names[:test_count]}
    groups.update({name: "val" for name in names[test_count:test_count + val_count]})
    groups.update({name: "train" for name in names[test_count + val_count:]})
    partitions = {name: [] for name in RECORDING_PARTITIONS}
    missing_features = []
    for (source, recording_id), recording in sorted(recordings.items()):
        for speaker, listener in recording["sessions"]:
            if verify_files:
                for sequence in (speaker, listener):
                    for modality in ("audio-features", "facial-attributes", "coefficients"):
                        path = (Path(root_dir) / source / modality / sequence).with_suffix(".npy")
                        if not path.is_file():
                            missing_features.append(str(path))
            for input_path, target_path in ((speaker, listener), (listener, speaker)):
                partitions[groups[recording_id]].append({
                    "source_split": source, "recording_id": recording_id,
                    "speaker_path": input_path, "listener_path": target_path,
                })
    if missing_features:
        raise FileNotFoundError("Missing {} required feature files; first: {}".format(
            len(missing_features), missing_features[0]))
    counts = validate_recording_partitions(partitions)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in RECORDING_PARTITIONS:
        path = output / (name + ".csv")
        write_csv(path, RECORDING_FIELDS, partitions[name])
        hashes[name] = file_digest(path)
    split_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
    audit = {
        "protocol": "recording_disjoint", "split_id": split_id, "split_seed": seed,
        "manifest_sha256": hashes, "counts": counts, "source_splits": list(source_splits),
        "test_fraction": test_fraction, "val_fraction": val_fraction,
        "files_verified": verify_files, "speaker_identity_disjoint": None,
        "listener_identity_disjoint": None,
        "recording_groups": {name: sorted(r for r, group in groups.items() if group == name)
                             for name in RECORDING_PARTITIONS},
        "scope": "Held-out recordings, not verified held-out people. Participant IDs are unavailable.",
    }
    (output / "split.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def load_split_bundle(split_dir):
    root = Path(split_dir)
    audit = json.loads((root / "split.json").read_text(encoding="utf-8"))
    protocol = audit.get("protocol")
    names = protocol_partitions(protocol)
    fields = PAIR_FIELDS if protocol == "speaker_independent" else RECORDING_FIELDS
    partitions = {}
    if set(audit["manifest_sha256"]) != set(names):
        raise ValueError("Manifest names do not match the split protocol")
    for name in names:
        path = root / (name + ".csv")
        if file_digest(path) != audit["manifest_sha256"][name]:
            raise ValueError("Manifest changed since preparation: {}".format(path))
        partitions[name] = read_csv(path, fields)
    split_id = hashlib.sha256(json.dumps(audit["manifest_sha256"], sort_keys=True).encode("utf-8")).hexdigest()
    if split_id != audit["split_id"]:
        raise ValueError("Split fingerprint mismatch")
    if protocol == "speaker_independent":
        counts = validate_partitions(partitions, audit["unseen_speaker_ids"])
    else:
        counts = validate_recording_partitions(partitions)
        expected_groups = {name: sorted({row["recording_id"] for row in rows})
                           for name, rows in partitions.items()}
        if audit.get("recording_groups") != expected_groups:
            raise ValueError("Recording assignment metadata does not match the manifests")
    if audit.get("counts") != counts:
        raise ValueError("Split counts do not match the manifests")
    return partitions, audit


def manifest_dataset_paths(manifest):
    manifest = Path(manifest)
    partitions, audit = load_split_bundle(manifest.parent)
    partition = manifest.stem
    if partition not in partitions:
        raise ValueError("Unknown experiment partition: {}".format(partition))
    rows = partitions[partition]
    speakers, listeners = [], []
    candidates = defaultdict(set)
    for row in rows:
        speaker = Path(row["source_split"]) / row["speaker_path"]
        listener = Path(row["source_split"]) / row["listener_path"]
        speakers.append(speaker)
        listeners.append(listener)
        candidates[tuple(listener.parts[1:3])].add(listener)
    ground_truths = [sorted(candidates[tuple(path.parts[1:3])]) for path in listeners]
    metadata = dict(audit, partition=partition)
    return speakers, listeners, ground_truths, rows, metadata


def verify_converter(checkpoint, split_dir):
    _, audit = load_split_bundle(split_dir)
    checkpoint = Path(checkpoint)
    proof = Path(str(checkpoint) + ".speaker_split.json")
    if not checkpoint.is_file() or not proof.is_file():
        raise FileNotFoundError("Require a converter trained on this split and its .speaker_split.json sidecar")
    provenance = json.loads(proof.read_text(encoding="utf-8"))
    if (provenance.get("split_id") != audit["split_id"]
            or provenance.get("protocol", "speaker_independent") != audit["protocol"]
            or provenance.get("checkpoint_sha256") != file_digest(checkpoint)
            or provenance.get("training_partition") != "train"):
        raise ValueError("Converter provenance does not match this isolated split/checkpoint")
    return provenance
