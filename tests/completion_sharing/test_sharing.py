"""Sharing checks: numerical archive, conflict refusal, and launcher boundaries."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('completion_bundle',ROOT/'scripts/completion_bundle.py')
bundle=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundle)


class SharingChecks(unittest.TestCase):
    def test_archive_checksum_tamper_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            manifest=json.loads((bundle.DEFAULT/'manifest.json').read_text())
            (p/'manifest.json').write_text(json.dumps(manifest))
            (p/'study-bundle.tar.gz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'checksum'):
                bundle.bundle(p)

    def test_conflicting_unpack_is_rejected_before_writes(self):
        old_root=bundle.ROOT
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            bundle.ROOT=Path(directory)
            conflict=bundle.ROOT/'results/study/b';conflict.parent.mkdir(parents=True)
            conflict.write_bytes(b'keep')
            try:
                with patch.object(bundle,'bundle',return_value=({}, {'results/study/a':b'new','results/study/b':b'changed'})):
                    with self.assertRaisesRegex(ValueError,'overwrite'):bundle.unpack(bundle.DEFAULT)
                self.assertEqual(conflict.read_bytes(),b'keep')
                self.assertFalse((conflict.parent/'a').exists())
            finally:bundle.ROOT=old_root

    def test_launcher_dry_run_has_full_cohort_and_no_state_creation(self):
        state=ROOT/'.session-runs/completion-reproduction'
        before={str(p):p.stat().st_mtime_ns for p in state.rglob('*')} if state.exists() else {}
        result=subprocess.run(['bash',str(ROOT/'scripts/run_completion_experiments.sh'),'--cohort',
            '--config','configs/task-driven-completion-cuda-v2.json','--python',sys.executable,'--dry-run'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.count('--task-index'),52)
        self.assertEqual(result.stdout.count('--summarize-only'),1)
        after={str(p):p.stat().st_mtime_ns for p in state.rglob('*')} if state.exists() else {}
        self.assertEqual(before,after)

    def test_launcher_rejects_incomplete_or_ambiguous_arguments(self):
        for args in [[],['--pilot','--cohort'],['--pilot','--config'],['--smoke','--config','configs/task-driven-completion-cuda-v2.json']]:
            result=subprocess.run(['bash',str(ROOT/'scripts/run_completion_experiments.sh'),*args],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0,args)


if __name__=='__main__':unittest.main()
