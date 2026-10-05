"""Rerun the legacy empty-output fixture without moving historical artifacts."""
import importlib.util,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('legacy_protocol_tests',ROOT/'tests/completion_transformer/test_protocol.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix='agfl-legacy-empty-output-fixture-') as directory:
 root=Path(directory);(root/'configs').mkdir()
 declaration=root/'configs/completion-transformer.json'
 declaration.write_bytes((ROOT/'configs/completion-transformer.json').read_bytes())
 # Only relocate this test's unchanged declaration/expected paths; the actual
 # loader and output-occupancy checks run unchanged. No historical output moves.
 with patch.object(module,'ROOT',root),patch.object(module,'DECLARATION',declaration):
  suite=unittest.TestSuite([module.ProtocolTests('test_fixed_declaration_plan_and_config_relative_paths')])
  result=unittest.TextTestRunner(verbosity=2).run(suite)
  if not result.wasSuccessful() or result.skipped: raise SystemExit(1)
print('PASS with byte-identical declaration in fresh temporary config/output fixture; original failure retained')
