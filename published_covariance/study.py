"""Exclusive task persistence and explicit synthetic/real cohort lifecycle."""
from __future__ import annotations
import hashlib, json, os, tempfile
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from .completion import CovarianceCompleter, zero_fill
from .data import load_native_recording, fit_native_normalization, normalize, stratified_split
from .masks import static_masks
from .training import train_covariance, _validation_record, select_checkpoint
from .protocol import ROOT, TASKS, ARMS, CONDITIONS, identity, sha256

_NORMALIZATION_CACHE={}

def _json_new(path, value):
    _safe_write_target(path)
    with path.open("x", encoding="utf8") as f: json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write("\n")

def _npz_new(path, **arrays):
    _safe_write_target(path)
    with path.open("xb") as f: np.savez_compressed(f, **arrays)

def _safe_write_target(path):
    path=Path(path)
    if path.is_symlink(): raise ValueError(f"refusing symlinked write target: {path}")
    for parent in path.parents:
        if parent.exists() and parent.is_symlink(): raise ValueError(f"refusing symlinked write path component: {parent}")
    if not path.parent.is_dir(): raise ValueError(f"write parent is absent or not a directory: {path.parent}")

def _hash(path): return sha256(path)

def _numeric_manifest(out):
    manifest={}
    for p in sorted(out.rglob("*")):
        if p.is_symlink(): raise ValueError(f"symlinked study artifact: {p}")
        if p.is_file() and p.name not in {"audit.json","report.json","report.md"}:
            manifest[str(p.relative_to(out))]=_hash(p)
    return manifest

def _safe_output(path: Path, synthetic: bool):
    if path.is_symlink(): raise ValueError("output root cannot be a symlink")
    absolute = path.absolute()
    if absolute == Path("/") or absolute == ROOT or absolute == ROOT / "results": raise ValueError("unsafe output root")
    for p in [absolute, *absolute.parents]:
        if p.exists() and p.is_symlink(): raise ValueError("symlinked output path component")
    if synthetic and (ROOT / "results") in absolute.parents: raise ValueError("synthetic studies cannot enter real output roots")
    if not synthetic and (absolute.parent != ROOT / "results" or not absolute.name.startswith("published-covariance-")): raise ValueError("real study output must be a direct results/published-covariance-* directory; choose a new suffix after identity changes")
    return absolute

def _safe_subdir(parent,name):
    p=parent/name
    if p.is_symlink(): raise ValueError(f"symlinked study subdirectory: {name}")
    p.mkdir(exist_ok=True)
    if not p.is_dir() or p.is_symlink(): raise ValueError(f"unsafe study subdirectory: {name}")
    return p

def _open_study(output, device, synthetic, *, create=True):
    out = _safe_output(Path(output), synthetic)
    ident = json.loads(json.dumps(identity(device, synthetic),sort_keys=True))
    if out.exists():
        if not out.is_dir() or out.is_symlink(): raise ValueError("output is not a safe directory")
        marker = out / "identity.json"
        if not marker.is_file() or marker.is_symlink(): raise ValueError("existing output is foreign or partial")
        if json.loads(marker.read_text()) != ident: raise ValueError("study identity changed; select a fresh output")
    else:
        if not create: raise ValueError("read-only mode requires an existing study identity")
        out.mkdir(parents=True, exist_ok=False)
        _json_new(out / "identity.json", ident)
    return out, ident

def _synthetic_subject(subject, session="T"):
    """Generated software fixture with label-dependent signals, never measurements."""
    rng = np.random.Generator(np.random.PCG64(int(subject[1:]) * 901 + 42))
    y = np.tile(np.arange(4, dtype=np.int32), 4)
    x = rng.normal(0, 0.3, size=(16, 22, 8)).astype(np.float32)
    for i, label in enumerate(y): x[i, label::4, :] += 0.5
    ids = tuple(f"{subject}:{session}:r99:t{i+1:03d}" for i in range(16))
    return x, y, ids

class _SyntheticClassifier:
    def __init__(self, subject):
        seed = int(subject[1:]) * 103
        rng = np.random.Generator(np.random.PCG64(seed))
        self.w = tf.constant(rng.normal(0, .3, size=(22, 4)).astype(np.float32))
    def __call__(self, x, training=False):
        z = tf.reduce_mean(x, axis=2)
        return tf.nn.softmax(tf.matmul(z, self.w), axis=-1)

def _classifier(subject, synthetic, assets, device="cpu"):
    if synthetic: return _SyntheticClassifier(subject)
    from .classifier import FrozenClassifier
    if assets is None: assets=ROOT/".session-runs/published-checkpoint-covariance/assets"
    with tf.device("/CPU:0" if device == "cpu" else "/GPU:0"):
        return FrozenClassifier(subject, assets_root=assets)

def _metrics(p, labels):
    p = np.asarray(p, dtype=np.float64); y = np.asarray(labels, dtype=np.int64); pred = p.argmax(axis=1)
    recalls = [np.mean(pred[y == c] == c) for c in range(4) if np.any(y == c)]
    return {"balanced_accuracy": float(np.mean(recalls)), "accuracy": float(np.mean(pred == y)), "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()), "n_trials": int(len(y))}

def _recording(subject, session, synthetic, assets):
    if synthetic: return _synthetic_subject(subject, session)
    assets = Path(assets) if assets is not None else ROOT / ".session-runs/published-checkpoint-covariance/assets"
    path = Path(assets) / "recordings" / f"{subject}{session}.mat"
    record = load_native_recording(path, subject, session)
    origin_path=Path(assets)/"recordings"/f"{subject}{session}-origin.json"
    origin=json.loads(origin_path.read_text())
    if origin.get("sha256")!=record.metadata["source_sha256"] or origin.get("bytes")!=path.stat().st_size: raise ValueError(f"recording bytes differ from pinned origin: {subject}{session}")
    # Published scaler is fit on every T trial, as it was in the checkpoint pipeline.
    tpath = Path(assets) / "recordings" / f"{subject}T.mat"
    t = load_native_recording(tpath, subject, "T")
    mean, scale = fit_native_normalization(t.values)
    _NORMALIZATION_CACHE[subject]=(mean.copy(),scale.copy(),tuple(t.trial_ids),t.metadata["source_sha256"])
    full = np.ones((len(record.values), 22), dtype=np.bool_)
    x = normalize(record.values, mean, scale, full).astype(np.float32)
    return x, record.labels.astype(np.int32), record.trial_ids

def _recording_source_sha(subject,session,assets):
    if assets is None: assets=ROOT/".session-runs/published-checkpoint-covariance/assets"
    return json.loads((Path(assets)/"recordings"/f"{subject}{session}-origin.json").read_text())["sha256"]

def _persist_normalization(out,subject,synthetic):
    if synthetic:return
    if subject not in _NORMALIZATION_CACHE: raise RuntimeError("native T normalization was not fit during data load")
    mean,scale,ids,source_sha=_NORMALIZATION_CACHE[subject]
    folder=_safe_subdir(out,"normalization"); path=folder/f"{subject}.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z["mean"],mean) or not np.array_equal(z["scale"],scale) or z["source_sha256"].item()!=source_sha or z["trial_ids"].astype(str).tolist()!=list(ids): raise ValueError("saved native normalization differs from T source")
    else:_npz_new(path,mean=mean,scale=scale,source_sha256=np.asarray(source_sha),trial_ids=np.asarray(ids),recipe=np.asarray("EEG-ATCNet shuffle(42), per-channel/timepoint StandardScaler fitted on all T trials"))

def _classifier_state_digest(classifier):
    model=getattr(classifier,"model",None)
    weights=model.weights if model is not None else [classifier.w]
    h=hashlib.sha256()
    for value in weights:
        arr=np.asarray(value.numpy())
        h.update(str(arr.dtype).encode());h.update(str(arr.shape).encode());h.update(arr.tobytes(order="C"))
    return h.hexdigest()

def _fit_one(out, subject, seed, device, synthetic, assets):
    if subject!="A01": _validate_pilot_review(out)
    folder = _safe_subdir(out,"fits")
    dest = folder / f"{subject}-s{seed}.npz"; meta = dest.with_suffix(".json")
    if dest.exists() or meta.exists():
        if not dest.is_file() or not meta.is_file() or _hash(dest) != json.loads(meta.read_text()).get("npz_sha256"):
            raise ValueError(f"existing task is partial or corrupt: {subject}/{seed}")
        _audit_fit_task(out,subject,seed,synthetic,assets,device)
        return
    x, y, ids = _recording(subject, "T", synthetic, assets)
    _persist_normalization(out,subject,synthetic)
    tr, va = stratified_split(y, seed)
    classifier = _classifier(subject, synthetic, assets, device)
    classifier_state_before=_classifier_state_digest(classifier)
    with tf.device("/CPU:0" if device == "cpu" else "/GPU:0"):
        fit = train_covariance(classifier, x[tr], y[tr], [ids[i] for i in tr], x[va], y[va], [ids[i] for i in va], subject, seed,
                               max_epochs=2 if synthetic else 100, min_epochs=1 if synthetic else 10, patience=1 if synthetic else 20, synthetic=synthetic)
    train_masks = np.stack([static_masks(subject, seed, [ids[i] for i in tr], "train", count, repeat=0, epoch=e) for e in range(0, fit["epochs_completed"] + 1) for count in (16, 6)])
    valid_masks=np.stack([static_masks(subject,seed,[ids[i] for i in va],"validation",count,repeat=repeat,epoch=0) for count in (16,6) for repeat in range(5)])
    seed_payload=json.dumps(["published-covariance-train-v1",subject,int(seed)],separators=(",",":")).encode()
    local_seed=int.from_bytes(hashlib.sha256(seed_payload).digest()[:16],"little")
    schedule_rng=np.random.Generator(np.random.PCG64(local_seed)); orders=[]; counts_by_epoch=[]
    for _ in range(fit["epochs_completed"]):
        order=schedule_rng.permutation(len(tr)); draw=np.concatenate([schedule_rng.choice(np.array([22,16,6],dtype=np.int32),size=min(32,len(order)-start)) for start in range(0,len(order),32)])
        orders.append(order); counts_by_epoch.append(draw)
    history = json.dumps(fit["history"], sort_keys=True, separators=(",", ":"))
    classifier_state_after=_classifier_state_digest(classifier)
    if classifier_state_before != classifier_state_after: raise RuntimeError("frozen classifier state changed during covariance fitting")
    train64=x[tr].astype(np.float64,copy=False); second=np.einsum("nct,ndt->cd",train64,train64,optimize=True)/(len(train64)*train64.shape[2])
    probe=tf.convert_to_tensor(x[va[:min(4,len(va))]])
    probe_mask=np.stack([static_masks(subject,seed,[ids[i]],"validation",16,repeat=0,epoch=0)[0] for i in va[:len(probe)]])
    selected_completer=CovarianceCompleter(second,trainable=False);selected_completer.free.assign(fit["selected_free"])
    _=selected_completer.complete(probe,probe_mask).numpy()
    t0=time.perf_counter();completed=selected_completer.complete(probe,probe_mask).numpy();completion_seconds=time.perf_counter()-t0
    _=classifier(completed,training=False).numpy();t0=time.perf_counter();_=classifier(completed,training=False).numpy();classifier_seconds=time.perf_counter()-t0
    _npz_new(dest, train_x=x[tr], train_y=y[tr], train_ids=np.asarray([ids[i] for i in tr]), validation_x=x[va], validation_y=y[va], validation_ids=np.asarray([ids[i] for i in va]), all_trial_ids=np.asarray(ids), all_labels=y, train_indices=tr, validation_indices=va, initial_free=fit["initial_free"], selected_free=fit["selected_free"], train_masks=train_masks, validation_masks=valid_masks, epoch_orders=np.asarray(orders,dtype=np.int64), retained_counts=np.asarray(counts_by_epoch,dtype=np.int32), history=np.asarray(history), selected_epoch=np.asarray(fit["selected_epoch"], dtype=np.int64))
    source_sha="synthetic-generated-v1" if synthetic else _recording_source_sha(subject,"T",assets)
    _json_new(meta, {"subject": subject, "seed": seed, "source_recording_sha256":source_sha,"train_ids_sha256": hashlib.sha256("\n".join(ids[i] for i in tr).encode()).hexdigest(), "validation_ids_sha256": hashlib.sha256("\n".join(ids[i] for i in va).encode()).hexdigest(), "selected_epoch": fit["selected_epoch"], "epochs_completed": fit["epochs_completed"], "classifier_state_sha256_before":classifier_state_before,"classifier_state_sha256_after":classifier_state_after,"classifier_parameter_count":int(sum(np.prod(v.shape) for v in (classifier.model.weights if hasattr(classifier,"model") else [classifier.w]))), "measured_probe_completion_seconds":completion_seconds,"measured_probe_classifier_seconds":classifier_seconds,"measured_probe_trial_count":len(probe), "npz_sha256": _hash(dest), "synthetic_fixture": synthetic})

def _fit_seal(out):
    tasks = []
    ident=json.loads((out/"identity.json").read_text())
    for subject, seed in TASKS:
        p = out / "fits" / f"{subject}-s{seed}.npz"; m = p.with_suffix(".json")
        if p.is_symlink() or m.is_symlink() or not p.is_file() or not m.is_file(): return None
        metadata=json.loads(m.read_text())
        if metadata.get("npz_sha256") != _hash(p) or metadata.get("subject")!=subject or metadata.get("seed")!=seed or metadata.get("synthetic_fixture")!=ident.get("synthetic_fixture"): return None
        with np.load(p,allow_pickle=False) as z:
            if z["initial_free"].shape!=(253,) or z["selected_free"].shape!=(253,) or not np.isfinite(z["selected_free"]).all(): return None
        ap=out/"fit-audits"/f"{subject}-s{seed}.json"
        if ap.is_symlink() or not ap.is_file(): return None
        receipt=json.loads(ap.read_text())
        if receipt!={"schema":1,"identity_sha256":_hash(out/"identity.json"),"subject":subject,"seed":seed,"fit_sha256":_hash(p),"passed":True}: return None
        tasks.append({"subject": subject, "seed": seed, "sha256": _hash(p), "metadata_sha256": _hash(m)})
    return {"schema": 1, "task_count": 27, "identity_sha256":_hash(out/"identity.json"), "tasks": tasks}

def mode_pilot(output, device, synthetic, assets):
    out, _ = _open_study(output, device, synthetic)
    for seed in (0, 1, 2): _fit_one(out, "A01", seed, device, synthetic, assets)
    return out

def mode_review_pilot(output,device="cpu",assets=None):
    out = Path(output).absolute(); tasks=[]
    with (out/"identity.json").open() as f: is_synthetic=json.load(f).get("synthetic_fixture",False)
    for seed in (0,1,2):
        p=out/"fits"/f"A01-s{seed}.npz"; m=p.with_suffix(".json")
        if not p.is_file() or not m.is_file() or json.loads(m.read_text()).get("npz_sha256") != _hash(p): raise ValueError("pilot is incomplete or corrupt")
        _record_fit_audit(out,"A01",seed,is_synthetic,assets,device)
        tasks.append({"task":p.name,"sha256":_hash(p)})
    review = {"identity_sha256": _hash(out/"identity.json"), "pilot_tasks":tasks,"accepted":True}
    p=out/"pilot-review.json"
    if p.exists():
        if json.loads(p.read_text()) != review: raise ValueError("pilot review does not match pilot bytes")
    else: _json_new(p,review)

def _validate_pilot_review(out):
    p=out/"pilot-review.json"
    if not p.is_file() or p.is_symlink(): raise ValueError("independent pilot review is required before fitting remaining participants")
    review=json.loads(p.read_text())
    tasks=[{"task":(out/"fits"/f"A01-s{s}.npz").name,"sha256":_hash(out/"fits"/f"A01-s{s}.npz")} for s in (0,1,2)]
    if review.get("identity_sha256")!=_hash(out/"identity.json") or review.get("pilot_tasks")!=tasks or review.get("accepted") is not True: raise ValueError("pilot review is forged, stale, or detached from the fit identity")

def mode_train_cohort(output, device, synthetic, assets):
    out=Path(output).absolute()
    _validate_pilot_review(out)
    for subject,seed in TASKS:
        _fit_one(out,subject,seed,device,synthetic,assets)
    for subject,seed in TASKS: _record_fit_audit(out,subject,seed,synthetic,assets,device)
    seal=_fit_seal(out)
    if seal is None: raise ValueError("cannot seal incomplete/corrupt fit cohort")
    p=out/"fit-seal.json"
    if p.exists():
        if json.loads(p.read_text()) != seal: raise ValueError("fit seal differs; output identity is incompatible")
    else: _json_new(p,seal)

def _load_fit(out, subject, seed):
    p=out/"fits"/f"{subject}-s{seed}.npz"; m=p.with_suffix(".json")
    if not p.is_file() or not m.is_file() or json.loads(m.read_text()).get("npz_sha256") != _hash(p): raise ValueError("missing/corrupt fit task")
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}

def _audit_fit_task(out,subject,seed,synthetic,assets,device):
    fit=_load_fit(out,subject,seed)
    tx,ty,tids=_recording(subject,"T",synthetic,assets)
    task_meta=json.loads((out/"fits"/f"{subject}-s{seed}.json").read_text())
    if task_meta.get("classifier_state_sha256_before") != task_meta.get("classifier_state_sha256_after"): raise ValueError("frozen classifier state changed during covariance fitting")
    if not isinstance(task_meta.get("classifier_parameter_count"),int) or task_meta["classifier_parameter_count"]<1: raise ValueError("classifier parameter count is missing")
    if not all(np.isfinite(task_meta.get(k,np.nan)) and task_meta[k]>=0 for k in ("measured_probe_completion_seconds","measured_probe_classifier_seconds")): raise ValueError("measured completion/classifier timings are missing")
    expected_source="synthetic-generated-v1" if synthetic else _recording_source_sha(subject,"T",assets)
    if task_meta.get("source_recording_sha256")!=expected_source: raise ValueError("task provenance is detached from native T bytes")
    if not synthetic:
        norm_path=out/"normalization"/f"{subject}.npz"
        if not norm_path.is_file() or norm_path.is_symlink(): raise ValueError("saved native T normalization is missing")
        with np.load(norm_path,allow_pickle=False) as z:
            cached=_NORMALIZATION_CACHE.get(subject)
            if cached is None or not np.array_equal(z["mean"],cached[0]) or not np.array_equal(z["scale"],cached[1]) or z["source_sha256"].item()!=cached[3] or z["trial_ids"].astype(str).tolist()!=list(cached[2]): raise ValueError("saved normalization differs from exact native T fit")
    tr,va=stratified_split(ty,seed)
    train_ids=[tids[i] for i in tr]; valid_ids=[tids[i] for i in va]
    if not np.array_equal(fit["train_indices"],tr) or not np.array_equal(fit["validation_indices"],va): raise ValueError("saved split indices differ from declared stratified split")
    if fit["train_ids"].astype(str).tolist()!=train_ids or fit["validation_ids"].astype(str).tolist()!=valid_ids or fit["all_trial_ids"].astype(str).tolist()!=list(tids) or not np.array_equal(fit["all_labels"],ty): raise ValueError("saved T trial identities/order differ")
    if not np.array_equal(fit["train_x"],tx[tr]) or not np.array_equal(fit["validation_x"],tx[va]) or not np.array_equal(fit["train_y"],ty[tr]) or not np.array_equal(fit["validation_y"],ty[va]): raise ValueError("persisted T fit arrays differ from pinned source")
    x64=fit["train_x"].astype(np.float64,copy=False); second=np.einsum("nct,ndt->cd",x64,x64,optimize=True)/(len(x64)*x64.shape[2])
    expected=CovarianceCompleter(second,trainable=False).free.numpy()
    if not np.array_equal(expected,fit["initial_free"]): raise ValueError("covariance initialization is not train-only second moment")
    if fit["selected_free"].shape!=(253,) or not np.isfinite(fit["selected_free"]).all(): raise ValueError("invalid selected covariance state")
    history=json.loads(str(fit["history"].item()))
    if len(history)!=int(json.loads((out/"fits"/f"{subject}-s{seed}.json").read_text())["epochs_completed"])+1: raise ValueError("training history is incomplete")
    if int(select_checkpoint(history)["epoch"]) != int(fit["selected_epoch"]): raise ValueError("selected epoch does not follow declared validation tie rule")
    epochs=int(fit["selected_epoch"])
    expected_masks=[]
    for epoch in range(0,len(history)):
        for retained in (16,6): expected_masks.append(static_masks(subject,seed,train_ids,"train",retained,repeat=0,epoch=epoch))
    if not np.array_equal(fit["train_masks"],np.stack(expected_masks)): raise ValueError("saved training mask bank differs from declared key schedule")
    valid_expected=np.stack([static_masks(subject,seed,valid_ids,"validation",count,repeat=repeat,epoch=0) for count in (16,6) for repeat in range(5)])
    if not np.array_equal(fit["validation_masks"],valid_expected): raise ValueError("saved validation mask bank differs from declared key schedule")
    seed_payload=json.dumps(["published-covariance-train-v1",subject,int(seed)],separators=(",",":")).encode(); local_seed=int.from_bytes(hashlib.sha256(seed_payload).digest()[:16],"little")
    rng=np.random.Generator(np.random.PCG64(local_seed)); orders=[]; counts=[]
    for _ in range(len(history)-1):
        order=rng.permutation(len(train_ids)); draw=np.concatenate([rng.choice(np.array([22,16,6],dtype=np.int32),size=min(32,len(order)-start)) for start in range(0,len(order),32)])
        orders.append(order);counts.append(draw)
    if not np.array_equal(fit["epoch_orders"],np.asarray(orders,dtype=np.int64)) or not np.array_equal(fit["retained_counts"],np.asarray(counts,dtype=np.int32)): raise ValueError("persisted training order/count schedule differs from local PCG64 stream")
    if int(fit["selected_epoch"])==0 and not np.array_equal(fit["selected_free"],fit["initial_free"]): raise ValueError("epoch-zero selection did not restore exact initial state")
    classifier=_classifier(subject,synthetic,assets,device)
    current_state=_classifier_state_digest(classifier)
    if current_state!=task_meta.get("classifier_state_sha256_before") or current_state!=task_meta.get("classifier_state_sha256_after"): raise ValueError("frozen classifier state hash does not match the current classifier")
    valid=tf.convert_to_tensor(fit["validation_x"])
    init=CovarianceCompleter(second,trainable=False)
    initial_record=_validation_record(classifier,init,valid,fit["validation_y"],valid_ids,subject,seed,0)
    for field in ("degraded_ba","degraded_log_loss"):
        if not np.isclose(initial_record[field],history[0][field],rtol=0,atol=1e-12): raise ValueError("epoch-zero validation replay differs")
    selected=CovarianceCompleter(second,trainable=False); selected.free.assign(fit["selected_free"])
    selected_record=_validation_record(classifier,selected,valid,fit["validation_y"],valid_ids,subject,seed,epochs)
    saved=history[epochs]
    for field in ("degraded_ba","degraded_log_loss"):
        if not np.isclose(selected_record[field],saved[field],rtol=0,atol=1e-12): raise ValueError("selected T-validation replay differs")

def _record_fit_audit(out,subject,seed,synthetic,assets,device):
    _audit_fit_task(out,subject,seed,synthetic,assets,device)
    folder=_safe_subdir(out,"fit-audits")
    fit=out/"fits"/f"{subject}-s{seed}.npz"
    receipt={"schema":1,"identity_sha256":_hash(out/"identity.json"),"subject":subject,"seed":seed,"fit_sha256":_hash(fit),"passed":True}
    p=folder/f"{subject}-s{seed}.json"
    if p.exists():
        if p.is_symlink() or json.loads(p.read_text())!=receipt: raise ValueError("existing fit audit receipt differs")
    else:_json_new(p,receipt)

def _predict_arm(classifier, x, y, ids, subject, seed, arm, condition, repeat, free=None, initial=None, save_masks=False):
    retained={"full_22":22,"static_16":16,"static_6":6}[condition]
    mask=static_masks(subject,seed,ids,"E",retained,repeat=repeat,epoch=0)
    if arm=="zero": pred=zero_fill(tf.convert_to_tensor(x),tf.convert_to_tensor(mask)).numpy()
    elif retained==22: pred=x
    else:
        c=CovarianceCompleter(_second_from_free(initial),trainable=False)
        c.free.assign(initial if arm=="covariance_fixed" else free)
        pred=c.complete(x,mask).numpy()
    probs=classifier(pred,training=False).numpy()
    return probs,mask

def _second_from_free(free):
    # Construct a valid seed matrix; assigned state fully determines replay covariance.
    return np.eye(22,dtype=np.float64)

def mode_reference_task(output,subject,device,synthetic,assets):
    out=Path(output).absolute()
    refdir=_safe_subdir(out,"references")
    if subject not in {f"A{i:02d}" for i in range(1,10)}: raise ValueError("invalid reference subject")
    p=refdir/f"{subject}-T.npz"
    x,y,ids=_recording(subject,"T",synthetic,assets); c=_classifier(subject,synthetic,assets,device)
    wrapper=c(x,training=False).numpy()
    if synthetic: native=wrapper.copy()
    else:
        from .native_reference import upstream_probabilities
        native=upstream_probabilities(subject,x,assets or ROOT/".session-runs/published-checkpoint-covariance/assets",device)
        if not np.allclose(native,wrapper,atol=1e-5,rtol=1e-4) or not np.array_equal(native.argmax(1),wrapper.argmax(1)): raise RuntimeError(f"published native source parity failed for {subject} T reference")
    arrays={"x":x,"y":y,"ids":np.asarray(ids),"probabilities":native,"wrapper_probabilities":wrapper,"metrics":np.asarray(json.dumps(_metrics(native,y))),"session":np.asarray("T"),"parity":np.asarray(True)}
    if p.exists():
        with np.load(p,allow_pickle=False) as z:
            if any(not np.array_equal(z[k],v) for k,v in arrays.items()): raise ValueError(f"existing native T reference differs: {subject}")
    else:_npz_new(p,**arrays)
    auditdir=_safe_subdir(out,"reference-audits")
    receipt={"schema":1,"identity_sha256":_hash(out/"identity.json"),"subject":subject,"reference_sha256":_hash(p),"native_source_replay":not synthetic,"passed":True}
    rp=auditdir/f"{subject}-T.json"
    if rp.exists():
        if rp.is_symlink() or json.loads(rp.read_text())!=receipt: raise ValueError("existing native T reference receipt differs")
    else:_json_new(rp,receipt)

def mode_reference(output,device,synthetic,assets):
    for subject in [f"A{i:02d}" for i in range(1,10)]: mode_reference_task(output,subject,device,synthetic,assets)

def mode_evaluate(output,device,synthetic,assets):
    out=Path(output).absolute()
    for subject,seed in TASKS: mode_evaluate_task(out,subject,seed,device,synthetic,assets)
    for subject,seed in TASKS: mode_audit_evaluation_task(out,subject,seed,device,synthetic,assets)
    aggregate_evaluation_tasks(out)

def _write_npz_immutable(path,arrays):
    if path.exists():
        if path.is_symlink(): raise ValueError(f"symlinked saved artifact: {path}")
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!=set(arrays) or any(not np.array_equal(z[k],v) for k,v in arrays.items()): raise ValueError(f"existing immutable artifact differs: {path.name}")
    else:_npz_new(path,**arrays)

def mode_evaluate_task(out,subject,seed,device,synthetic,assets):
    seal=out/"fit-seal.json"; expected=_fit_seal(out)
    if expected is None or not seal.is_file() or json.loads(seal.read_text())!=expected: raise ValueError("all 27 selected covariance states must be sealed before any E evaluation")
    fit_audit=out/"fit-audits"/f"{subject}-s{seed}.json"
    if not fit_audit.is_file(): raise ValueError("independent T fit audit required before E evaluation task")
    reports=out/"evaluation"
    taskdir=out/"evaluation-tasks"; taskpath=taskdir/f"{subject}-s{seed}.json"
    expected_cells=[reports/f"{subject}-s{seed}-{arm}-{condition}-r{repeat}.npz" for arm in ARMS for condition,nrep in CONDITIONS.items() for repeat in range(nrep)]
    auditpath=out/"evaluation-audits"/f"{subject}-s{seed}.json"
    if taskpath.is_symlink() or auditpath.is_symlink() or any(p.is_symlink() for p in expected_cells): raise ValueError("symlinked evaluation task artifact")
    present=[p.is_file() for p in expected_cells]
    if taskpath.exists():
        if not all(present): raise ValueError("completed evaluation task has missing cells; refusing partial repair")
        mode_audit_evaluation_task(out,subject,seed,device,synthetic,assets)
        return
    if auditpath.exists() or any(present): raise ValueError("partial evaluation task exists without its immutable task receipt; choose a fresh study output")
    t_ref=out/"references"/f"{subject}-T.npz"
    if synthetic and not t_ref.is_file(): mode_reference_task(out,subject,device,True,assets)
    if not t_ref.is_file(): raise ValueError("native T reference is required before E evaluation")
    if not synthetic:
        tr=out/"reference-audits"/f"{subject}-T.json"
        if not tr.is_file() or json.loads(tr.read_text()).get("reference_sha256")!=_hash(t_ref): raise ValueError("source-backed T reference audit is required before E evaluation")
    reports=_safe_subdir(out,"evaluation")
    x,y,ids=_recording(subject,"E",synthetic,assets); classifier=_classifier(subject,synthetic,assets,device)
    input_path=reports/f"{subject}-E-input.npz"
    _write_npz_immutable(input_path,{"native_x":x,"labels":y,"ids":np.asarray(ids),"source_sha256":np.asarray("synthetic-generated-v1" if synthetic else _recording_source_sha(subject,"E",assets)),"normalization_source_sha256":np.asarray("synthetic-generated-v1" if synthetic else _recording_source_sha(subject,"T",assets))})
    if synthetic: native_full=classifier(x,training=False).numpy()
    else:
        from .native_reference import upstream_probabilities
        native_full=upstream_probabilities(subject,x,assets or ROOT/".session-runs/published-checkpoint-covariance/assets",device)
        wrapped=classifier(x,training=False).numpy()
        if not np.allclose(native_full,wrapped,atol=1e-5,rtol=1e-4) or not np.array_equal(native_full.argmax(1),wrapped.argmax(1)): raise RuntimeError(f"published native E reference parity failed for {subject}")
    ref_arrays={"ids":np.asarray(ids),"labels":y,"probabilities":native_full,"metrics":np.asarray(json.dumps(_metrics(native_full,y))),"session":np.asarray("E"),"parity":np.asarray(True)}
    _write_npz_immutable(reports/f"{subject}-E-native-reference.npz",ref_arrays)
    fit=_load_fit(out,subject,seed); rows=[]
    for arm in ARMS:
        for condition,nrep in CONDITIONS.items():
            for repeat in range(nrep):
                probs,mask=_predict_arm(classifier,x,y,ids,subject,seed,arm,condition,repeat,fit["selected_free"],fit["initial_free"])
                m=_metrics(probs,y);key=f"{subject}-s{seed}-{arm}-{condition}-r{repeat}.npz"
                _write_npz_immutable(reports/key,{"ids":np.asarray(ids),"labels":y,"mask":mask,"probabilities":probs,"balanced_accuracy":np.asarray(m["balanced_accuracy"]),"accuracy":np.asarray(m["accuracy"]),"log_loss":np.asarray(m["log_loss"]),"n_trials":np.asarray(m["n_trials"])})
                rows.append({"subject":subject,"seed":seed,"arm":arm,"condition":condition,"repeat":repeat,**m})
    taskdir=_safe_subdir(out,"evaluation-tasks"); taskpath=taskdir/f"{subject}-s{seed}.json"
    receipt={"schema":1,"identity_sha256":_hash(out/"identity.json"),"subject":subject,"seed":seed,"rows":rows,"e_input_sha256":_hash(input_path),"e_native_reference_sha256":_hash(reports/f"{subject}-E-native-reference.npz"),"passed":True}
    if taskpath.exists():
        if taskpath.is_symlink() or json.loads(taskpath.read_text())!=receipt: raise ValueError("existing evaluation task receipt differs")
    else:_json_new(taskpath,receipt)

def mode_audit_evaluation_task(out,subject,seed,device,synthetic,assets):
    seal=out/"fit-seal.json"; expected_seal=_fit_seal(out)
    if expected_seal is None or not seal.is_file() or json.loads(seal.read_text())!=expected_seal: raise ValueError("complete 27-task fit seal required before E audit")
    taskpath=out/"evaluation-tasks"/f"{subject}-s{seed}.json"
    if taskpath.is_symlink() or not taskpath.is_file(): raise ValueError("evaluation task receipt missing")
    task=json.loads(taskpath.read_text())
    if task.get("identity_sha256")!=_hash(out/"identity.json") or task.get("subject")!=subject or task.get("seed")!=seed: raise ValueError("evaluation task receipt identity mismatch")
    x,y,ids=_recording(subject,"E",synthetic,assets); classifier=_classifier(subject,synthetic,assets,device)
    reports=out/"evaluation"; full_ref=None
    input_path=reports/f"{subject}-E-input.npz"
    with np.load(input_path,allow_pickle=False) as stored_input:
        if not np.array_equal(stored_input["native_x"],x) or not np.array_equal(stored_input["labels"],y) or not np.array_equal(stored_input["ids"].astype(str),np.asarray(ids)): raise ValueError("persisted E input differs from native recording")
        if not synthetic and (stored_input["source_sha256"].item()!=_recording_source_sha(subject,"E",assets) or stored_input["normalization_source_sha256"].item()!=_recording_source_sha(subject,"T",assets)): raise ValueError("E input recording/normalization provenance differs")
    refpath=reports/f"{subject}-E-native-reference.npz"
    if not refpath.is_file(): raise ValueError("native E reference missing")
    if not synthetic:
        from .native_reference import upstream_probabilities
        native=upstream_probabilities(subject,x,assets or ROOT/".session-runs/published-checkpoint-covariance/assets",device)
    else: native=classifier(x,training=False).numpy()
    with np.load(refpath,allow_pickle=False) as z:
        if z["session"].item()!="E" or not np.array_equal(z["ids"].astype(str),np.asarray(ids)) or not np.array_equal(z["labels"],y) or not np.array_equal(z["probabilities"],native): raise ValueError("native E reference failed replay")
    fit=_load_fit(out,subject,seed); seen=set()
    expected={(arm,cond,rep) for arm in ARMS for cond,nrep in CONDITIONS.items() for rep in range(nrep)}
    for r in task["rows"]:
        key=(r["arm"],r["condition"],r["repeat"])
        if key in seen or key not in expected: raise ValueError("evaluation task has duplicate or foreign cells")
        seen.add(key); path=reports/f"{subject}-s{seed}-{r['arm']}-{r['condition']}-r{r['repeat']}.npz"
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z["ids"].astype(str),np.asarray(ids)) or not np.array_equal(z["labels"],y): raise ValueError("saved E labels/IDs differ from official source")
            mask=static_masks(subject,seed,ids,"E",{"full_22":22,"static_16":16,"static_6":6}[r["condition"]],repeat=r["repeat"],epoch=0)
            if not np.array_equal(z["mask"],mask): raise ValueError("saved E mask differs from declared key")
            probs,_=_predict_arm(classifier,x,y,ids,subject,seed,r["arm"],r["condition"],r["repeat"],fit["selected_free"],fit["initial_free"])
            if not np.array_equal(z["probabilities"],probs): raise ValueError("saved E probabilities differ from independent replay")
            m=_metrics(probs,y)
            for field in ("balanced_accuracy","accuracy","log_loss"):
                if not np.isclose(float(z[field]),m[field],rtol=0,atol=1e-12) or not np.isclose(r[field],m[field],rtol=0,atol=1e-12): raise ValueError("saved E cell metrics differ from probability replay")
            if int(z["n_trials"])!=len(y): raise ValueError("saved E trial count differs")
            if r["condition"]=="full_22":
                if full_ref is not None and not np.array_equal(probs,full_ref): raise ValueError("full-input predictions differ across arms")
                full_ref=probs.copy()
    if seen!=expected: raise ValueError("evaluation task cell index is incomplete")
    auditdir=_safe_subdir(out,"evaluation-audits"); p=auditdir/f"{subject}-s{seed}.json"
    cell_hashes={f.name:_hash(f) for f in sorted(reports.glob(f"{subject}-s{seed}-*.npz"))}
    expected_cell_names={f"{subject}-s{seed}-{arm}-{condition}-r{repeat}.npz" for arm in ARMS for condition,nrep in CONDITIONS.items() for repeat in range(nrep)}
    if set(cell_hashes)!=expected_cell_names: raise ValueError("evaluation task byte manifest does not contain its exact expected cell set")
    receipt={"schema":1,"identity_sha256":_hash(out/"identity.json"),"subject":subject,"seed":seed,"task_sha256":_hash(taskpath),"cell_sha256":cell_hashes,"passed":True}
    receipt["e_input_sha256"]=_hash(input_path);receipt["e_native_reference_sha256"]=_hash(refpath)
    if p.exists():
        if p.is_symlink() or json.loads(p.read_text())!=receipt: raise ValueError("existing evaluation audit receipt differs")
    else:_json_new(p,receipt)

def aggregate_evaluation_tasks(out):
    rows=[]
    for subject,seed in TASKS:
        p=out/"evaluation-tasks"/f"{subject}-s{seed}.json"; a=out/"evaluation-audits"/f"{subject}-s{seed}.json"
        if not p.is_file() or not a.is_file(): raise ValueError("all 27 independently audited E tasks required for aggregation")
        audit=json.loads(a.read_text())
        if audit.get("identity_sha256")!=_hash(out/"identity.json") or audit.get("task_sha256")!=_hash(p) or audit.get("e_input_sha256")!=_hash(out/"evaluation"/f"{subject}-E-input.npz") or audit.get("e_native_reference_sha256")!=_hash(out/"evaluation"/f"{subject}-E-native-reference.npz"): raise ValueError("E task audit receipt or shared inputs are stale")
        rows.extend(json.loads(p.read_text())["rows"])
    result=out/"evaluation-rows.npz"
    _write_npz_immutable(result,{"json_rows":np.asarray(json.dumps(rows,sort_keys=True,separators=(",",":")))})

def mode_audit(output,device,synthetic,assets):
    out=Path(output).absolute(); sealp=out/"fit-seal.json"
    expected=_fit_seal(out)
    if expected is None or not sealp.is_file() or json.loads(sealp.read_text()) != expected: raise ValueError("fit seal absent or invalid")
    # Task receipts were independently replayed in bounded children. Aggregate
    # those receipts and hash the complete immutable byte tree without reopening
    # all 891 prediction arrays.
    fit_receipts=[out/"fit-audits"/f"{s}-s{k}.json" for s,k in TASKS]
    eval_receipts=[out/"evaluation-audits"/f"{s}-s{k}.json" for s,k in TASKS]
    if all(p.is_file() and not p.is_symlink() for p in fit_receipts+eval_receipts):
        rowsfile=out/"evaluation-rows.npz"
        if not rowsfile.is_file(): raise ValueError("evaluation task aggregation missing")
        with np.load(rowsfile,allow_pickle=False) as z: rows=json.loads(str(z["json_rows"].item()))
        expected_cells={(s,k,arm,cond,rep) for s,k in TASKS for arm in ARMS for cond,nrep in CONDITIONS.items() for rep in range(nrep)}
        seen=[(r.get("subject"),int(r.get("seed",-1)),r.get("arm"),r.get("condition"),int(r.get("repeat",-1))) for r in rows if isinstance(r,dict)]
        if len(seen)!=len(expected_cells) or set(seen)!=expected_cells: raise ValueError("aggregate evaluation cell index is incomplete, duplicate, or foreign")
        task_rows=[]; full_input={}
        for s,k in TASKS:
            taskpath=out/"evaluation-tasks"/f"{s}-s{k}.json"; task=json.loads(taskpath.read_text())
            if task.get("identity_sha256")!=_hash(out/"identity.json"): raise ValueError("evaluation task belongs to another identity")
            task_rows.extend(task.get("rows",[]))
            for arm in ARMS:
                cell=out/"evaluation"/f"{s}-s{k}-{arm}-full_22-r0.npz"
                with np.load(cell,allow_pickle=False) as z: prob=z["probabilities"].copy()
                if s in full_input and not np.array_equal(full_input[s],prob): raise ValueError("full-input predictions differ across arms or covariance seeds")
                full_input[s]=prob
        key=lambda r:(r["subject"],int(r["seed"]),r["arm"],r["condition"],int(r["repeat"]))
        if sorted(task_rows,key=key)!=sorted(rows,key=key): raise ValueError("aggregate evaluation rows or metrics differ from audited task receipts")
        for receipt_path in fit_receipts+eval_receipts:
            rec=json.loads(receipt_path.read_text())
            if rec.get("identity_sha256")!=_hash(out/"identity.json") or rec.get("passed") is not True: raise ValueError("task audit receipt identity mismatch")
            if "fit_sha256" in rec:
                fp=out/"fits"/f"{rec['subject']}-s{rec['seed']}.npz"
                if _hash(fp)!=rec["fit_sha256"]: raise ValueError("fit changed after independent audit")
            if "cell_sha256" in rec:
                taskpath=out/"evaluation-tasks"/f"{rec['subject']}-s{rec['seed']}.json"
                if _hash(taskpath)!=rec.get("task_sha256"): raise ValueError("evaluation task receipt changed after audit")
                if rec.get("e_input_sha256")!=_hash(out/"evaluation"/f"{rec['subject']}-E-input.npz") or rec.get("e_native_reference_sha256")!=_hash(out/"evaluation"/f"{rec['subject']}-E-native-reference.npz"): raise ValueError("shared E input/native reference changed after task audit")
                expected_names={f"{rec['subject']}-s{rec['seed']}-{arm}-{condition}-r{repeat}.npz" for arm in ARMS for condition,nrep in CONDITIONS.items() for repeat in range(nrep)}
                if set(rec["cell_sha256"])!=expected_names: raise ValueError("task audit receipt does not bind the exact expected cell set")
                for name,digest in rec["cell_sha256"].items():
                    if _hash(out/"evaluation"/name)!=digest: raise ValueError("evaluation cell changed after independent audit")
        if not synthetic:
            for i in range(1,10):
                s=f"A{i:02d}"; ref=out/"references"/f"{s}-T.npz"; rp=out/"reference-audits"/f"{s}-T.json"
                if not ref.is_file() or not rp.is_file(): raise ValueError("source-backed T reference or its audit receipt is missing")
                receipt=json.loads(rp.read_text())
                if receipt.get("identity_sha256")!=_hash(out/"identity.json") or receipt.get("reference_sha256")!=_hash(ref) or receipt.get("native_source_replay") is not True: raise ValueError("T reference audit receipt failed identity or byte binding")
        result={"passed":True,"schema":2,"identity_sha256":_hash(out/"identity.json"),"fit_tasks":27,"evaluated_cells":len(rows),"evaluation_complete":True,"synthetic_fixture":synthetic,"replayed_saved_metrics":True,"numeric_artifacts":_numeric_manifest(out),"checkpoint_provenance_grade":"B","prior_project_E_outcome_exposure":"unknown"}
        p=out/"audit.json"
        if p.exists():
            if p.is_symlink() or json.loads(p.read_text())!=result: raise ValueError("existing audit differs from independently replayed receipts or numeric byte manifest")
        else:_json_new(p,result)
        return
    raise ValueError("bounded per-task fit and evaluation audit receipts are incomplete; refusing an unbounded whole-cohort replay")
    for subject,seed in TASKS: _audit_fit_task(out,subject,seed,synthetic,assets,device)
    if not synthetic:
        from .native_reference import upstream_probabilities
        root=assets or ROOT/".session-runs/published-checkpoint-covariance/assets"
        for i in range(1,10):
            subject=f"A{i:02d}"
            p=out/"references"/f"{subject}-T.npz"
            if not p.is_file(): raise ValueError("native full-input T reference is missing")
            x,y,ids=_recording(subject,"T",False,assets); upstream=upstream_probabilities(subject,x,root,device)
            with np.load(p,allow_pickle=False) as z:
                if z["session"].item()!="T" or not np.array_equal(z["ids"].astype(str),np.asarray(ids)) or not np.array_equal(z["y"],y) or not np.array_equal(z["probabilities"],upstream): raise ValueError("saved T reference differs from unmodified pinned source")
    rowsfile=out/"evaluation-rows.npz"
    if rowsfile.is_file():
        with np.load(rowsfile,allow_pickle=False) as z: rows=json.loads(str(z["json_rows"].item()))
        expected={(s,seed,arm,cond,rep) for s,seed in TASKS for arm in ARMS for cond,nrep in CONDITIONS.items() for rep in range(nrep)}
        seen=set()
        if not isinstance(rows,list): raise ValueError("evaluation row index must be a list")
        for r in rows:
            if not isinstance(r,dict): raise ValueError("malformed evaluation row index")
            try: key=(r["subject"],int(r["seed"]),r["arm"],r["condition"],int(r["repeat"]))
            except (KeyError,TypeError,ValueError): raise ValueError("evaluation row index has invalid key")
            if key in seen: raise ValueError(f"duplicate evaluation cell index: {key}")
            if key not in expected: raise ValueError(f"foreign evaluation cell index: {key}")
            seen.add(key)
        if seen != expected: raise ValueError(f"evaluation cell index is incomplete: {len(seen)}/{len(expected)}")
        full_reference={}; replay_inputs={}; classifiers={}; fits={}
        for r in rows:
            p=out/"evaluation"/f"{r['subject']}-s{r['seed']}-{r['arm']}-{r['condition']}-r{r['repeat']}.npz"
            if r["subject"] not in replay_inputs:
                src=out/"evaluation"/f"{r['subject']}-E-input.npz"
                if not src.is_file(): raise ValueError("saved native E input missing")
                with np.load(src,allow_pickle=False) as z: replay_inputs[r["subject"]]=(z["native_x"].copy(),z["labels"].copy(),z["ids"].astype(str).tolist())
                if not synthetic:
                    with np.load(src,allow_pickle=False) as z:
                        if z["source_sha256"].item()!=_recording_source_sha(r["subject"],"E",assets) or z["normalization_source_sha256"].item()!=_recording_source_sha(r["subject"],"T",assets): raise ValueError("E input manifest differs from pinned data/normalization")
                native=_recording(r["subject"],"E",synthetic,assets)
                if not np.array_equal(native[0],replay_inputs[r["subject"]][0]) or not np.array_equal(native[1],replay_inputs[r["subject"]][1]) or list(native[2])!=replay_inputs[r["subject"]][2]: raise ValueError("persisted E identity differs from pinned native recordings")
                classifiers[r["subject"]]=_classifier(r["subject"],synthetic,assets,device)
                if not synthetic:
                    from .native_reference import upstream_probabilities
                    reference_path=out/"evaluation"/f"{r['subject']}-E-native-reference.npz"
                    if not reference_path.is_file(): raise ValueError("pinned-source E full-input reference is missing")
                    native_probs=upstream_probabilities(r["subject"],native[0],assets or ROOT/".session-runs/published-checkpoint-covariance/assets",device)
                    with np.load(reference_path,allow_pickle=False) as native_saved:
                        if not np.array_equal(native_saved["labels"],native[1]) or not np.array_equal(native_saved["ids"].astype(str),np.asarray(native[2])) or not np.array_equal(native_saved["probabilities"],native_probs): raise ValueError("saved E reference differs from unmodified pinned source")
            with np.load(p,allow_pickle=False) as z:
                probs=z["probabilities"]; labels=z["labels"]; x,source_y,ids=replay_inputs[r["subject"]]; mask=z["mask"]
                if not np.array_equal(labels,source_y) or not np.array_equal(z["ids"].astype(str),np.asarray(ids)): raise ValueError("evaluation labels/trial IDs differ from native input")
                expected_mask=static_masks(r["subject"],r["seed"],ids,"E",{"full_22":22,"static_16":16,"static_6":6}[r["condition"]],repeat=r["repeat"],epoch=0)
                if not np.array_equal(mask,expected_mask): raise ValueError("saved E mask differs from declared key")
                fit_key=(r["subject"],r["seed"])
                if fit_key not in fits: fits[fit_key]=_load_fit(out,*fit_key)
                fit=fits[fit_key]
                replay,_=_predict_arm(classifiers[r["subject"]],x,labels,ids,r["subject"],r["seed"],r["arm"],r["condition"],r["repeat"],fit["selected_free"],fit["initial_free"])
                if not np.array_equal(replay,probs): raise ValueError("independent probability replay differs")
                m=_metrics(probs,labels)
                for name in ("balanced_accuracy","accuracy","log_loss"):
                    if not np.isclose(m[name],r[name],rtol=0,atol=1e-12): raise ValueError(f"saved metric replay failed: {p.name}/{name}")
                    if name not in r: raise ValueError(f"evaluation row lacks {name}: {p.name}")
                    if not np.isclose(float(z[name]),m[name],rtol=0,atol=1e-12): raise ValueError(f"saved cell metric replay failed: {p.name}/{name}")
                if int(z["n_trials"]) != m["n_trials"]: raise ValueError(f"saved cell trial count differs: {p.name}")
                if not np.isfinite(probs).all() or np.any(probs<0) or not np.allclose(probs.sum(1),1,atol=1e-5): raise ValueError("invalid saved probabilities")
                if r["condition"]=="full_22":
                    k=r["subject"]
                    if k in full_reference and not np.array_equal(full_reference[k],probs): raise ValueError("full-input prediction changed across arms")
                    full_reference[k]=probs.copy()
    artifacts=_numeric_manifest(out)
    result={"passed":True,"schema":2,"identity_sha256":_hash(out/"identity.json"),"fit_tasks":27,"evaluated_cells":len(rows) if rowsfile.is_file() else 0,"evaluation_complete":rowsfile.is_file(),"synthetic_fixture":synthetic,"replayed_saved_metrics":rowsfile.is_file(),"numeric_artifacts":artifacts,"checkpoint_provenance_grade":"B","prior_project_E_outcome_exposure":"unknown"}
    p=out/"audit.json"
    if p.exists():
        if json.loads(p.read_text())!=result: raise ValueError("existing audit differs from replay")
    else:_json_new(p,result)

def verify_audit_artifacts(out):
    p=out/"audit.json"
    if not p.is_file() or p.is_symlink(): raise ValueError("a successful identity-bound audit is required before reporting")
    evidence=json.loads(p.read_text())
    if evidence.get("passed") is not True or evidence.get("identity_sha256")!=_hash(out/"identity.json"): raise ValueError("audit evidence is absent, incomplete, or belongs to another identity")
    if evidence.get("evaluation_complete") is not True or evidence.get("evaluated_cells")!=891: raise ValueError("complete independently audited evaluation is required for cohort reporting")
    expected=evidence.get("numeric_artifacts")
    if not isinstance(expected,dict) or not expected: raise ValueError("audit lacks a complete numeric artifact manifest")
    if _numeric_manifest(out)!=expected: raise ValueError("audited numeric artifact bytes or inventory changed")
    return evidence
