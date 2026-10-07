import unittest
from published_covariance.reporting import summarize_cells

class ReportingTests(unittest.TestCase):
    def test_incomplete_suppresses_primary_and_discloses_exposure(self):
        result=summarize_cells([])
        self.assertFalse(result["complete"])
        self.assertNotIn("cohort",result)
        self.assertEqual(result["checkpoint_provenance_grade"],"B")
        self.assertEqual(result["prior_project_E_outcome_exposure"],"unknown")
    def test_duplicates_are_rejected(self):
        row={"subject":"A01","seed":0,"arm":"zero","condition":"full_22","repeat":0,"balanced_accuracy":.5,"accuracy":.5,"log_loss":1.,"n_trials":4}
        with self.assertRaises(ValueError): summarize_cells([row,row])

if __name__=="__main__": unittest.main(verbosity=2)
