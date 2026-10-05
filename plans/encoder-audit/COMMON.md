# Instructions for each encoder investigation session

Implement exactly the current session. The user approved an investigation of
class information and training behavior, not a large architecture search.
Read `AGENTS.md`, this plan's `CONTRACT.md`, the current task, and supplied
handoffs. Inspect Git status. Preserve all prior work, including uncommitted
LaTeX/PDF documentation. No subagents or nested session runners.

1. Use `AGFL_PYTHON` for checks. Do not install packages, download recordings,
   run real fits, or start GPU jobs. Sessions 01–07 implement and test tools;
   a human runs the real audit afterward. Session 08 reads verified evidence.
2. Edit only the task's bounded paths. Necessary repairs to an earlier module
   in `inm/encoder_audit/` are permitted with explanation and regression checks.
   Do not edit legacy/baseline research code, completed outputs, existing configs,
   this plan, automation scripts, or `tests/automation/`. No commits or pushes.
3. Missing numerical dependencies block a numeric session. Missing real data
   should be handled explicitly by the implementation and does not block its
   synthetic tests. Missing real evidence blocks session 08. Never turn missing
   data, a checksum mismatch, skipped tests, or incomplete coverage into success.
   This applies to the operation being validated: an inventory must report
   incompatible real inputs as blocked, while software acceptance can pass when
   the session implements and tests that failure correctly. Do not bypass the
   real-input failure or report that replay is ready.
4. Use tiny CPU `unittest` fixtures. No GDF files, network, paid calls, or real
   research fitting in automated tests. Include meaningful failure fixtures:
   leakage, mismatched identities, reordered samples, and nonfinite inputs.
5. Follow the stable APIs and JSON contracts. Keep imports lazy for plan and
   inventory operations. Do not create placeholder modules for later sessions.
6. Run the session tests and any affected earlier tests. Review the diff and
   retain original input bytes. Existing validation maxima are selection-biased;
   test metrics must never enter diagnostic decisions or generated audit views.
7. Return exactly the supplied JSON schema, with this session ID, `completed`
   or `blocked`, changed files, actual checks, blockers, and concise API handoff.
   The runner verifies completion independently. Partial useful work should be
   preserved and reported as blocked; do not claim the whole task is complete.
   For `completed`, provide a nonempty summary, actual executed checks, and
   `blockers: []`. Reserve final-response blockers for unfinished work or failed
   acceptance in the current session. Preserve downstream input-readiness
   findings in the inventory, documentation, and `next_session_notes`; these
   findings alone do not block sessions 01–07 when their bounded implementation
   and required synthetic checks are complete. Session 08 requires real evidence.

Coding receipts establish software acceptance only, not research success. A
negative or inconclusive scientific result is valid. Avoid claims that linear
probe failure proves absence of information or that an NRMSE threshold proves
a representation is adequate for classification.
