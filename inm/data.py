"""Original run-filtered BCI2a trials, with mask-aware raw outage views."""
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from eeg_models.datasets.base import SignalDataset, bandpass_finite_spans, normalize_samples, source_fingerprint
from .availability import CHANNEL_IDS


@dataclass
class AvailabilityDataset(SignalDataset):
    # Runtime context is separate from the original full-input dataset identity.
    # Source bytes, trial IDs and preprocessing already identify these recordings.
    recordings: dict = field(default_factory=dict, repr=False)
    trial_contexts: list = field(default_factory=list, repr=False)
    _observed_cache: dict = field(default_factory=dict, init=False, repr=False)

    def observed_signals(self, indices, mask, window_samples):
        """Filter observed raw spans, never filtering across a synthetic outage.

        Outages apply within the cue-aligned trial. Raw context outside that
        interval remains observed, as in the original offline run filtering.
        Uninterrupted electrodes reuse the original filtered signal. A partial
        electrode is refiltered from raw data with missing intervals set to NaN.
        Cache by trial/electrode/schedule so all models share the same input.
        """
        mask = np.asarray(mask)
        samples = self.metadata['samples']
        if (mask.dtype != np.bool_ or samples % window_samples
                or mask.shape != (len(indices), 22, samples // window_samples)
                or not mask.any(axis=1).all()):
            raise ValueError('Expected Boolean [N,22,P] masks with observations in every window')
        output = np.zeros((len(indices), 22, samples), dtype=np.float32)
        settings = self.metadata['preprocessing']
        for row, index in enumerate(indices):
            context = self.trial_contexts[index]
            recording, valid = self.recordings[context['recording']]
            start, stop = context['start'], context['start'] + samples
            left, right = context['run_start'], context['run_stop']
            for channel, schedule in enumerate(mask[row]):
                if schedule.all():
                    output[row, channel] = self.x[index, channel]
                    continue
                if not schedule.any():
                    continue
                key = (int(index), channel, tuple(schedule.tolist()))
                if key not in self._observed_cache:
                    available = valid[left:right].copy()
                    available[start - left:stop - left] &= np.repeat(schedule, window_samples)
                    # Select before filtering: hidden raw values are not operands
                    # of the filter. Native recording gaps also remain excluded.
                    observed = np.where(available[None, :], recording[channel:channel + 1, left:right], np.nan)
                    filtered, _ = bandpass_finite_spans(observed, self.metadata['sampling_rate'],
                                                       settings['lowcut'], settings['highcut'])
                    trial = filtered[0, start - left:stop - left]
                    sample_mask = np.repeat(schedule, window_samples)
                    if not np.isfinite(trial[sample_mask]).all():
                        raise ValueError(f'Observed span too short to filter in {self.sample_ids[index]}, channel {channel}')
                    self._observed_cache[key] = np.where(sample_mask, trial, 0.).astype(np.float32)
                output[row, channel] = self._observed_cache[key]
        return output


def load_subject(config):
    import mne
    from scipy.io import loadmat

    root = Path(config["data_dir"]).expanduser().resolve()
    subjects = config["subjects"]
    sessions = [str(s).upper() for s in config["sessions"]]
    if not subjects or any(type(s) is not int or s < 1 or s > 9 for s in subjects) or len(set(subjects)) != len(subjects):
        raise ValueError("BCI IV 2a subjects must be a nonempty unique subset of 1..9")
    if not sessions or len(set(sessions)) != len(sessions) or not set(sessions) <= {"T", "E"}:
        raise ValueError("BCI IV 2a sessions must be a unique nonempty subset of ['T', 'E']")
    if config["artifact_policy"] not in {"include", "exclude"}:
        raise ValueError("artifact_policy must be include or exclude")
    if config["filter_scope"] != "run":
        raise ValueError("This baseline uses the original run-level filtering")
    window = config["window"]
    if type(window) is not int or window < 1 or not np.isfinite(config['offset_seconds']) or config["offset_seconds"] < 0:
        raise ValueError("EEG window must be positive and offset_seconds nonnegative")
    signals, labels, groups, ids, sample_sessions, sample_runs = [], [], [], [], [], []
    sources, skipped = [], {"artifact": 0, "out_of_bounds": 0, "nonfinite": 0}
    fs_values, artifact_count = set(), 0
    recordings, trial_contexts = {}, []
    for subject in sorted(subjects):
        for session in sorted(sessions):
            basename = f"A{subject:02d}{session}"
            path = root / f"{basename}.gdf"
            if not path.is_file():
                raise FileNotFoundError(f"Required EEG recording not found: {path}; set data.data_dir explicitly")
            sources.append(source_fingerprint(path))
            raw = mne.io.read_raw_gdf(str(path), preload=True, verbose="ERROR")
            fs = float(raw.info["sfreq"])
            fs_values.add(fs)
            if len(raw.ch_names) < 22:
                raise ValueError(f"{path.name} has fewer than 22 EEG channels")
            continuous = raw.get_data()[:22].astype(np.float64)
            events, event_dict = mne.events_from_annotations(raw, verbose="ERROR")
            names = {value: key for key, value in event_dict.items()}
            coded_events = [(int(event[0]) - int(raw.first_samp), names[event[2]]) for event in events]
            run_starts = sorted({position for position, code in coded_events if code == "32766"})
            trial_starts = sorted({position for position, code in coded_events if code == "768"})
            rejected = [position for position, code in coded_events if code == "1023"]
            cues = [(position, code) for position, code in coded_events if code in {"769", "770", "771", "772", "783"}]
            external_labels = None
            if any(code == "783" for _, code in cues):
                labels_root = Path(config["labels_dir"]).expanduser() if config["labels_dir"] else root
                labels_path = labels_root / f"{basename}.mat"
                if not labels_path.is_file():
                    raise FileNotFoundError(
                        f"{path.name} contains unknown evaluation cues (783). Supply official {basename}.mat "
                        "with classlabel via data.labels_dir; never infer test labels from cue codes.")
                official = loadmat(labels_path)
                if "classlabel" not in official:
                    raise ValueError(f"{labels_path} must contain official 'classlabel' labels")
                values = np.asarray(official["classlabel"]).reshape(-1)
                if len(values) != len(cues) or not np.isin(values, [1, 2, 3, 4]).all():
                    raise ValueError(f"{labels_path} must contain one label in 1..4 per cue ({len(cues)} cues)")
                external_labels = values.astype(np.int64) - 1
                sources.append(source_fingerprint(labels_path))
            original = continuous
            continuous, valid = bandpass_finite_spans(continuous, fs, config['lowcut'],
                                                       config['highcut'], run_starts, gdf_missing=True)
            recordings[basename] = (original, valid)
            offset = int(round(float(config["offset_seconds"]) * fs))
            for cue_index, (position, code) in enumerate(cues):
                label = int(external_labels[cue_index]) if external_labels is not None else int(code) - 769
                if label not in range(4):
                    raise ValueError(f"Unresolved class label in {path.name} at sample {position}")
                trial_index = bisect_right(trial_starts, position) - 1
                trial_start = trial_starts[trial_index] if trial_index >= 0 else position
                trial_end = trial_starts[trial_index + 1] if trial_index + 1 < len(trial_starts) else continuous.shape[1]
                artifact = any(trial_start <= marker < trial_end for marker in rejected)
                artifact_count += int(artifact)
                if artifact and config["artifact_policy"] == "exclude":
                    skipped["artifact"] += 1
                    continue
                start, stop = position + offset, position + offset + window
                next_run = bisect_right(run_starts, position)
                run_end = run_starts[next_run] if next_run < len(run_starts) else continuous.shape[1]
                if start < 0 or stop > min(trial_end, run_end, continuous.shape[1]):
                    skipped["out_of_bounds"] += 1
                    continue
                epoch = continuous[:, start:stop]
                if not np.isfinite(epoch).all():
                    skipped["nonfinite"] += 1
                    continue
                signals.append(epoch.astype(np.float32))
                labels.append(label)
                groups.append(f"A{subject:02d}")
                ids.append(f"{basename}:cue:{cue_index:03d}:sample:{position}")
                sample_sessions.append(session)
                sample_runs.append(f"{basename}:run:{bisect_right(run_starts, position) - 1}")
                run_start = run_starts[next_run - 1] if next_run else 0
                trial_contexts.append({'recording': basename, 'start': start,
                                       'run_start': run_start, 'run_stop': run_end})
    if not signals:
        raise ValueError(f"No usable EEG trials loaded from {root}; skipped={skipped}")
    if len(fs_values) != 1:
        raise ValueError("EEG recordings have different sample rates; explicit resampling is required")
    x = normalize_samples(np.stack(signals), config["normalization"])
    return AvailabilityDataset(x, np.asarray(labels), np.asarray(groups), ids, {
        "dataset": "eeg", "modality": "eeg", "num_classes": 4, "sampling_rate": fs_values.pop(),
        "channel_names": list(CHANNEL_IDS), "label_names": ["left_hand", "right_hand", "feet", "tongue"],
        "preprocessing": config, "sources": sources, "skipped": skipped,
        "artifact_trials_seen": artifact_count, "sample_sessions": sample_sessions,
        "sample_runs": sample_runs, "protocol_version": "bci2a-v2",
        "offline_zero_phase_filter": True,
    }, recordings=recordings, trial_contexts=trial_contexts)
