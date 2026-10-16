"""Train and evaluate FlowREACT on held-out recordings without inventing person IDs."""

from run_speaker_independent import main


if __name__ == "__main__":
    main(protocol="recording_disjoint")
