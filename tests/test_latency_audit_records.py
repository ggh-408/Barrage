import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_latency_evaluation import preserved_records


class AuditRecordsTests(unittest.TestCase):
    def test_reused_pid_with_identical_prediction_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folders = [root / 'audit_boundaries/completed_020',
                       root / 'audit_boundaries/completed_040', root / 'worker_checks']
            for folder, tracks in zip(folders, (100, 100, 101)):
                folder.mkdir(parents=True)
                (folder / 'worker_42.json').write_text(json.dumps(dict(calls=10, tracks=tracks)))
                (folder / 'prediction_42.json').write_text(json.dumps(dict(hint_calls=10)))
            for prefix in ('worker', 'prediction'):
                values = preserved_records(root, prefix + '_*.json',
                    list(folders[-1].glob(prefix + '_*.json')))
                self.assertEqual(len(values), 2)
                key = 'calls' if prefix == 'worker' else 'hint_calls'
                self.assertEqual(sum(item[key] for item in values), 20)

    def test_missing_companion_cannot_pass_as_complete_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'prediction_42.json'
            path.write_text('{"hint_calls": 10}')
            with self.assertRaises(FileNotFoundError):
                preserved_records(root, 'prediction_*.json', [path])


if __name__ == '__main__':
    unittest.main()
