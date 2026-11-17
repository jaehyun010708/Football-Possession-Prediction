# Source / attribution

- Refactored from the attached `more-than-a-hack-2027-submission-main.zip` (user-provided).
- `third_party/pa_sm/`: copied from the ZIP's `third_party/pa_sm/`, originally contributed in the V1 repository noted by the ZIP README.
- `third_party/models_new.py`: copied from the ZIP's `third_party/models_new.py`, originally from `mtah-e4` as documented by the ZIP README.
- `ad_tree/features_original.py` and `ad_tree/config.py`: copied from the ZIP's `preprocess/` feature generator and config; `features_original.py` import path updated for packaging.
- `preprocessing/event_rules.py`: copied from the ZIP's `preprocess/relabel_controlled_possession.py` (event relabeling heuristic).
- `ball_missing/mask_generator.py`: adapted from the user's synthetic ball gap code with the same binned ATD statistics and 10.6%-target. **Not exact line-for-line source.**
- IDSSE original open dataset: PySport `pysport/idsse-data`, licensed CC-BY-4.0.

Retain author attribution and any licenses that apply to reused components before redistributing the project. The absence of an upstream license file in the user-supplied ZIP does not grant a new license to copied code.
