import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from orbit import campaign


class CampaignContracts(unittest.TestCase):
    def test_utonia_sequences_training_and_full_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'reports').mkdir()
            for name in ('aligned-utonia-v1','utonia-xyz-v1','spatial-v1'):
                path=root/'cache'/name
                path.mkdir(parents=True)
                (path/'COMPLETE.json').write_text('{}')
            with patch.object(campaign,'ROOT',root),patch('sys.argv',['campaign','--mode','utonia']),\
                 patch.object(campaign.subprocess,'run') as run:
                campaign.main()
            commands=[call.args[0] for call in run.call_args_list]
            modules=[command[3] for command in commands]
            self.assertEqual(modules[:3],['orbit.train_linear_selector','orbit.train_masks','orbit.apex_variant'])
            self.assertEqual(modules.count('orbit.evaluate'),4)
            self.assertLess(modules.index('orbit.candidate_bank'),len(modules)-2)
            for command in commands:
                self.assertNotIn('--limit',command)
                if command[3]=='orbit.evaluate':self.assertIn('--posterior',command)
                if command[3]=='orbit.candidate_bank':self.assertIn('--spatial',command)
            journal=json.loads((root/'reports/campaign-utonia.json').read_text())
            self.assertEqual(journal['status'],'complete')
            self.assertFalse(journal['test_used'])

    def test_failed_stage_stops_dependents(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'reports').mkdir()
            with patch.object(campaign,'ROOT',root),patch('sys.argv',['campaign','--mode','sonata-checks']),\
                 patch.object(campaign.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'test')) as run:
                with self.assertRaises(subprocess.CalledProcessError):campaign.main()
            self.assertEqual(run.call_count,1)
            journal=json.loads((root/'reports/campaign-sonata-checks.json').read_text())
            self.assertEqual(journal['status'],'failed')


if __name__=='__main__':unittest.main()
