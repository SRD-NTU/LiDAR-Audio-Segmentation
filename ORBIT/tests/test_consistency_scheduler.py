import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'scripts/consistency_watch.py'
spec=importlib.util.spec_from_file_location('consistency_watch',path)
watch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class SchedulerContracts(unittest.TestCase):
    def test_completed_jobs_are_not_sent_back_as_slurm_dependencies(self):
        with patch.object(watch,'output',return_value='101|COMPLETED|0:0|\n102|RUNNING|0:0|'):
            self.assertEqual(watch.active_dependencies([101,102]),[102])

    def test_all_completed_dependencies_allow_immediate_submission(self):
        with patch.object(watch,'output',return_value='101|COMPLETED|0:0|'):
            self.assertEqual(watch.active_dependencies([101]),[])

    def test_failed_or_unknown_dependencies_are_not_bypassed(self):
        for record in ('101|FAILED|1:0|','101|COMPLETED|1:0|',''):
            with self.subTest(record=record),patch.object(watch,'output',return_value=record):
                with self.assertRaises(RuntimeError):watch.active_dependencies([101])


if __name__=='__main__':unittest.main()
