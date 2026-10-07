"""Independent checks for API preservation and declared provenance coverage."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

class IdentityAcceptance(unittest.TestCase):
    def test_original_package_api(self):
        import published_covariance as package
        from published_covariance import data
        names=('CHANNEL_IDS','CLASS_NAMES','NativeRecording','fit_native_normalization',
               'load_native_recording','load_normalization','normalize','save_normalization','stratified_split')
        for name in names: self.assertIs(getattr(package,name),getattr(data,name))

    def test_identity_covers_implementation_launcher_and_declaration(self):
        from published_covariance.protocol import identity
        value=identity('cpu',True)
        serialized=json.dumps(value,sort_keys=True)
        required=[*sorted((ROOT/'published_covariance').glob('*.py')),
                  ROOT/'scripts/run_published_covariance.sh',
                  ROOT/'plans/published-checkpoint-covariance/CONTRACT.md',
                  ROOT/'docs/published-covariance/checkpoint-audit.json',
                  ROOT/'configs/published-covariance.json']
        for path in required:
            self.assertIn(hashlib.sha256(path.read_bytes()).hexdigest(),serialized,str(path))
        # These packages alter the graph, matrix algebra or GPU kernels too.
        packages=value['runtime']['packages']
        normalized={k.lower().replace('_','-'):v for k,v in packages.items()}
        for name in ('tensorflow','keras','numpy','scipy','scikit-learn','h5py','nvidia-cudnn-cu12','nvidia-cublas-cu12'):
            self.assertEqual(normalized[name],importlib.metadata.version(name))

if __name__=='__main__': unittest.main(verbosity=2)
