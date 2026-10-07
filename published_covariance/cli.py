"""Explicit command line for the published checkpoint covariance study."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import sys

MODES=("plan","preflight","smoke","reference","reference-task","pilot","review-pilot","train-cohort","evaluate-cohort","evaluate-task","aggregate-evaluation","audit","audit-fit-task","audit-evaluation-task","summarize","fit-task","seal-cohort","mark-pilot")

def plan():
    tasks=[{"subject":f"A{i:02d}","seed":s} for i in range(1,10) for s in range(3)]
    return {"covariance_fits":27,"backbone_fits":0,"arm_evaluations":81,"tasks":tasks}

def main(argv=None):
    ap=argparse.ArgumentParser(prog="python -m published_covariance")
    ap.add_argument("mode",choices=MODES)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    ap.add_argument("--synthetic",action="store_true",help="isolated generated-data software fixture")
    ap.add_argument("--native",action="store_true",help="run the published native-checkpoint synthetic CUDA smoke")
    ap.add_argument("--assets",type=Path)
    ap.add_argument("--subject",choices=tuple(f"A{i:02d}" for i in range(1,10)))
    ap.add_argument("--seed",type=int)
    args=ap.parse_args(argv)
    if args.mode=="plan": print(json.dumps(plan(),sort_keys=True)); return 0
    if args.mode=="smoke" and not (args.synthetic or args.native): raise SystemExit("smoke requires --synthetic or --native")
    if args.mode=="smoke" and args.synthetic and args.native: raise SystemExit("choose one smoke fixture")
    if args.mode not in {"preflight","smoke","plan"} and not args.synthetic and args.device != "cuda:0": raise RuntimeError("real fitting/scoring requires explicit --device cuda:0; CPU fallback is forbidden")
    if args.mode in {"pilot","train-cohort","evaluate-cohort","evaluate-task","aggregate-evaluation","reference","reference-task","summarize","audit","audit-fit-task","audit-evaluation-task","fit-task","seal-cohort","mark-pilot"} and args.output is None: raise SystemExit(f"{args.mode} requires --output PATH")
    if args.mode=="smoke" and args.synthetic and args.output is None: raise SystemExit("synthetic smoke requires --output PATH")
    if args.mode=="smoke" and args.native and args.output is None: raise SystemExit("native smoke requires an exclusive --output PATH")
    if args.mode=="review-pilot" and args.output is None: raise SystemExit("review-pilot requires --output PATH")
    from .protocol import ROOT, identity
    from .study import _open_study, _safe_write_target, mode_pilot, mode_review_pilot, mode_train_cohort, mode_evaluate, mode_reference, mode_reference_task, mode_audit
    from .reporting import summarize_cells
    if args.device=="cuda:0":
        import tensorflow as tf
        if not tf.config.list_physical_devices("GPU"): raise RuntimeError("CUDA was requested but TensorFlow exposes no GPU; CPU fallback is forbidden")
        for gpu in tf.config.list_physical_devices("GPU"):
            try: tf.config.experimental.set_memory_growth(gpu,True)
            except RuntimeError as exc: raise RuntimeError("CUDA memory-growth policy must be set before model/device initialization") from exc
    expected_assets=ROOT/".session-runs/published-checkpoint-covariance/assets"
    if not args.synthetic and args.assets is not None and args.assets.resolve()!=expected_assets.resolve(): raise RuntimeError("--assets must resolve to the pinned audited asset bundle")
    if args.mode=="smoke" and args.native:
        if args.device!="cuda:0": raise RuntimeError("native-checkpoint smoke requires --device cuda:0")
        if args.output is None: raise SystemExit("native smoke requires an exclusive --output PATH")
        from .native_smoke import run_native_smoke
        metadata=run_native_smoke(args.output,args.assets or expected_assets,args.device)
        print(json.dumps({"native_checkpoint_smoke":"passed",**metadata},sort_keys=True));return 0
    if args.mode=="preflight":
        from .protocol import ROOT
        assets=args.assets or ROOT/".session-runs/published-checkpoint-covariance/assets"
        if not args.synthetic:
            if args.device!="cuda:0": raise RuntimeError("real preflight requires --device cuda:0")
            from .classifier import FrozenClassifier
            # Native forward/backward gate uses one generated input and one audited checkpoint.
            import numpy as np, tensorflow as tf
            with tf.device("/GPU:0"):
                model=FrozenClassifier("A01",assets_root=assets)
                x=tf.Variable(np.zeros((1,22,1125),dtype=np.float32))
                with tf.GradientTape() as tape: p=model(x); loss=tf.reduce_sum(p[:,0])
                g=tape.gradient(loss,x)
            if g is None or not np.isfinite(g.numpy()).all() or model.model.trainable_variables: raise RuntimeError("native synthetic input-gradient/frozen-weight gate failed")
            from .study import _recording
            for i in range(1,10):
                for session in ("T","E"): _recording(f"A{i:02d}",session,False,assets)
        print(json.dumps({"ready":True,"synthetic":args.synthetic,"runtime":identity(args.device,args.synthetic)["runtime"]},sort_keys=True)); return 0
    if args.mode=="review-pilot":
        _open_study(args.output,args.device,args.synthetic,create=False); mode_review_pilot(args.output,args.device,args.assets); return 0
    if args.mode=="fit-task":
        if args.output is None or args.subject is None or args.seed not in (0,1,2): raise SystemExit("fit-task requires --output, --subject and --seed")
        out,_=_open_study(args.output,args.device,args.synthetic)
        from .study import _fit_one
        _fit_one(out,args.subject,args.seed,args.device,args.synthetic,args.assets); return 0
    if args.mode=="audit-fit-task":
        if args.output is None or args.subject is None or args.seed not in (0,1,2): raise SystemExit("audit-fit-task requires --output, --subject and --seed")
        out,_=_open_study(args.output,args.device,args.synthetic,create=False)
        from .study import _record_fit_audit
        _record_fit_audit(out,args.subject,args.seed,args.synthetic,args.assets,args.device);return 0
    if args.mode=="reference-task":
        if args.output is None or args.subject is None: raise SystemExit("reference-task requires --output and --subject")
        out,_=_open_study(args.output,args.device,args.synthetic)
        mode_reference_task(out,args.subject,args.device,args.synthetic,args.assets);return 0
    if args.mode in {"evaluate-task","audit-evaluation-task"}:
        if args.output is None or args.subject is None or args.seed not in (0,1,2): raise SystemExit(f"{args.mode} requires --output, --subject and --seed")
        out,_=_open_study(args.output,args.device,args.synthetic,create=(args.mode=="evaluate-task"))
        from .study import mode_evaluate_task, mode_audit_evaluation_task
        if args.mode=="evaluate-task": mode_evaluate_task(out,args.subject,args.seed,args.device,args.synthetic,args.assets)
        else: mode_audit_evaluation_task(out,args.subject,args.seed,args.device,args.synthetic,args.assets)
        return 0
    if args.mode=="aggregate-evaluation":
        out,_=_open_study(args.output,args.device,args.synthetic,create=False)
        from .study import aggregate_evaluation_tasks
        aggregate_evaluation_tasks(out);return 0
    if args.mode=="mark-pilot":
        _open_study(args.output,args.device,args.synthetic)
        from .study import _load_fit, _record_fit_audit
        for seed in (0,1,2):
            _load_fit(args.output,"A01",seed)
            _record_fit_audit(args.output,"A01",seed,args.synthetic,args.assets,args.device)
        from .study import _json_new, _hash
        marker={"identity_sha256":_hash(args.output/"identity.json"),"tasks":[_hash(args.output/"fits"/f"A01-s{s}.npz") for s in (0,1,2)]}
        marker_path=args.output/"pilot-complete.json"
        if marker_path.exists():
            if marker_path.is_symlink() or json.loads(marker_path.read_text())!=marker: raise ValueError("existing pilot completion marker differs")
        else:_json_new(marker_path,marker)
        return 0
    if args.mode=="seal-cohort":
        _open_study(args.output,args.device,args.synthetic)
        from .study import _fit_seal, _json_new
        seal=_fit_seal(args.output)
        if seal is None: raise RuntimeError("cannot seal incomplete/corrupt fit cohort")
        p=args.output/"fit-seal.json"
        if p.exists():
            if json.loads(p.read_text()) != seal: raise RuntimeError("existing fit seal differs")
        else:_json_new(p,seal)
        return 0
    if args.mode=="pilot": mode_pilot(args.output,args.device,args.synthetic,args.assets); return 0
    if args.mode=="smoke": mode_pilot(args.output,args.device,True,args.assets); return 0
    if args.mode=="train-cohort":
        _open_study(args.output,args.device,args.synthetic); mode_train_cohort(args.output,args.device,args.synthetic,args.assets); return 0
    if args.mode=="reference":
        _open_study(args.output,args.device,args.synthetic); mode_reference(args.output,args.device,args.synthetic,args.assets); return 0
    if args.mode=="evaluate-cohort":
        _open_study(args.output,args.device,args.synthetic,create=False); mode_evaluate(args.output,args.device,args.synthetic,args.assets); return 0
    if args.mode=="audit":
        _open_study(args.output,args.device,args.synthetic,create=False); mode_audit(args.output,args.device,args.synthetic,args.assets); return 0
    if args.mode=="summarize":
        _open_study(args.output,args.device,args.synthetic,create=False)
        from .study import verify_audit_artifacts
        audit_evidence=verify_audit_artifacts(args.output)
        p=args.output/"evaluation-rows.npz"
        if not p.is_file(): raise RuntimeError("evaluation rows are absent")
        import numpy as np
        with np.load(p,allow_pickle=False) as z: rows=json.loads(str(z["json_rows"].item()))
        result=summarize_cells(rows)
        if not result["complete"]: raise RuntimeError("audited evaluation rows do not form a complete cohort")
        result["disclosures"]={"data_kind":"synthetic generated fixture" if args.synthetic else "official EEG-ATCNet native T/E recordings", "checkpoint_provenance_grade":"B", "prior_project_E_outcome_exposure":"unknown", "external_checkpoint_E_selection_excluded":False, "covariance_seeds_are_pretrained_backbone_seeds":False, "native_backbone_and_normalization_saw_all_T_trials_including_covariance_validation":not args.synthetic, "interpretation":"Exploratory descriptive published-checkpoint robustness; no confirmatory or independent generalization claim."}
        result["audit_identity_sha256"]=audit_evidence["identity_sha256"]
        fit_rows=[]
        for subject in (f"A{i:02d}" for i in range(1,10)):
            for seed in (0,1,2):
                meta=json.loads((args.output/"fits"/f"{subject}-s{seed}.json").read_text())
                with np.load(args.output/"fits"/f"{subject}-s{seed}.npz",allow_pickle=False) as z:
                    history=json.loads(str(z["history"].item())); selected=int(z["selected_epoch"]);t_ids=z["all_trial_ids"].astype(str).tolist()
                fit_rows.append({"subject":subject,"seed":seed,"selected_epoch":selected,"epoch_zero_selected":selected==0,"epoch_budget_saturated":meta["epochs_completed"]>=100,"validation_balanced_accuracy":history[selected]["degraded_ba"],"validation_log_loss":history[selected]["degraded_log_loss"],"classifier_parameter_count":meta.get("classifier_parameter_count"),"covariance_parameter_count":253,"classifier_state_sha256_before":meta.get("classifier_state_sha256_before"),"classifier_state_sha256_after":meta.get("classifier_state_sha256_after"),"measured_probe_completion_seconds":meta.get("measured_probe_completion_seconds"),"measured_probe_classifier_seconds":meta.get("measured_probe_classifier_seconds"),"measured_probe_trial_count":meta.get("measured_probe_trial_count"),"t_trial_ids":t_ids})
        result["fit_selection"]={"tasks":fit_rows,"covariance_parameter_count":253,"epoch_zero_selections":sum(x["epoch_zero_selected"] for x in fit_rows),"epoch_budget_saturations":sum(x["epoch_budget_saturated"] for x in fit_rows),"native_T_normalization_includes_covariance_validation":not args.synthetic,"covariance_seeds_are_pretrained_backbone_seeds":False}
        result["native_trial_identity"]={}
        for i in range(1,10):
            subject=f"A{i:02d}"; result["native_trial_identity"][subject]={}
            for sess in ("T","E"):
                if sess=="T":
                    if (args.output/"references"/f"{subject}-T.npz").is_file():
                        with np.load(args.output/"references"/f"{subject}-T.npz",allow_pickle=False) as z: ids=z["ids"].astype(str).tolist()
                    else: ids=fit_rows[(i-1)*3]["t_trial_ids"]
                else:
                    source=args.output/"evaluation"/f"{subject}-E-input.npz"
                    with np.load(source,allow_pickle=False) as z: ids=z["ids"].astype(str).tolist()
                result["native_trial_identity"][subject][sess]={"included_count":len(ids),"included_trial_ids":ids,"included_ids_sha256":__import__("hashlib").sha256("\n".join(ids).encode()).hexdigest(),"excluded_trial_ids":[]}
        # Machine and human report are created once; repeated summary is byte-stable.
        data=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
        report=args.output/"report.json"; md=args.output/"report.md"
        if report.exists() or md.exists():
            expected_md=render_markdown(result)
            if not report.is_file() or not md.is_file() or report.read_text()!=data or md.read_text()!=expected_md: raise RuntimeError("existing report differs from recomputed aggregation")
        else:
            _safe_write_target(report);_safe_write_target(md)
            with report.open("x") as f:f.write(data)
            text=render_markdown(result)
            with md.open("x") as f:f.write(text)
        print(data,end="");return 0
    return 2

def render_markdown(result):
    lines=["# Published checkpoint covariance report", "", result["disclosures"]["interpretation"], "",
           f"Data: {result['disclosures']['data_kind']}. Checkpoint grade B; prior project E-outcome exposure unknown; external E selection excluded: {result['disclosures']['external_checkpoint_E_selection_excluded']}.",
           "Covariance seeds vary covariance splits, training order, and masks; they are not independently pretrained backbone seeds.",
           "The published backbone and normalization used all T trials, including the later covariance-validation subset. No independent generalization or confirmatory claim is made.", "",
           "## Cohort metrics", "", "| Arm | Condition | BA | Accuracy | Log loss | BA degradation from full |", "|---|---|---:|---:|---:|---:|"]
    for arm in ("zero","covariance_fixed","covariance_learned"):
        for condition in ("full_22","static_16","static_6"):
            m=result["cohort"][arm]["conditions"][condition]
            lines.append(f"| {arm} | {condition} | {m['balanced_accuracy']:.6f} | {m['accuracy']:.6f} | {m['log_loss']:.6f} | {m['degradation_from_full_ba']:.6f} |")
    primary=result["primary_learned_minus_fixed"]
    lines += ["", f"Primary learned minus fixed mean: {primary['mean']:.6f} (participant bootstrap 95% interval {primary['bootstrap_95'][0]:.6f}, {primary['bootstrap_95'][1]:.6f}).", "",
              "## Participant effects", "", "| Participant | Learned − fixed | Learned − zero |", "|---|---:|---:|"]
    for subject,row in result["participants"].items(): lines.append(f"| {subject} | {row['learned_minus_fixed']:.6f} | {row['learned_minus_zero']:.6f} |")
    selection=result["fit_selection"]
    lines += ["", f"## Covariance checkpoint selection", "", f"Epoch zero selected: {selection['epoch_zero_selections']}/27. Epoch budget saturated: {selection['epoch_budget_saturations']}/27.", f"Each learned covariance has {selection['covariance_parameter_count']} free parameters. Completion latency and classifier inference latency below are separately measured for the same validation probe.", "", "| Participant | Seed | Epoch | Validation degraded BA | Validation degraded log loss | Completion seconds | Classifier seconds | Probe trials | Backbone parameters | Covariance parameters | State hash before = after |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for t in selection["tasks"]: lines.append(f"| {t['subject']} | {t['seed']} | {t['selected_epoch']} | {t['validation_balanced_accuracy']:.6f} | {t['validation_log_loss']:.6f} | {t['measured_probe_completion_seconds']:.6g} | {t['measured_probe_classifier_seconds']:.6g} | {t['measured_probe_trial_count']} | {t['classifier_parameter_count']} | {t['covariance_parameter_count']} | {t['classifier_state_sha256_before']==t['classifier_state_sha256_after']} |")
    lines += ["", "## Native trial inclusion identities", "", "| Participant | Session | Included trials | Included ID SHA256 | Excluded IDs |", "|---|---|---:|---|---|"]
    for subject,sessions in result["native_trial_identity"].items():
        for sess,meta in sessions.items(): lines.append(f"| {subject} | {sess} | {meta['included_count']} | `{meta['included_ids_sha256']}` | {', '.join(meta['excluded_trial_ids']) or 'none'} |")
    lines += ["", f"Audited study identity SHA256: `{result['audit_identity_sha256']}`.", ""]
    return "\n".join(lines)
