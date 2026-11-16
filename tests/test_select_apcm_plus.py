import unittest

from utils.select_apcm_plus import build_selection_rows
from utils.run_apcm_experiments import parse_run_output


def run(variant, seed, val_mse, test_mse, test_mae=0.2):
    return {
        'dataset': 'Solar',
        'pred_len': 336,
        'variant': variant,
        'seed': seed,
        'best_vali_mse': val_mse,
        'mse': test_mse,
        'mae': test_mae,
    }


class SelectApcmPlusTest(unittest.TestCase):
    def test_selects_by_validation_mse_not_test_mse(self):
        runs = [
            run('GTR-FD5L1', 2024, 0.30, 0.10),
            run('APCM-FD5L1', 2024, 0.20, 0.40),
            run('APCM-CE-FD5L1', 2024, 0.25, 0.05),
        ]

        rows, warnings = build_selection_rows(runs)

        self.assertEqual(warnings, [])
        self.assertEqual(rows[0]['chosen'], 'APCM-FD5L1')
        self.assertEqual(rows[0]['selection_metric'], 'vali_mse')
        self.assertAlmostEqual(rows[0]['mse'], 0.40)

    def test_scores_candidates_on_common_seeds_only(self):
        runs = [
            run('GTR-FD5L1', 2024, 0.30, 0.30),
            run('APCM-FD5L1', 2024, 0.25, 0.25),
            run('APCM-CE-FD5L1', 2024, 0.20, 0.20),
            run('APCM-CE-FD5L1', 2025, 9.00, 0.40),
            run('APCM-CE-FD5L1', 2026, 9.00, 0.60),
        ]

        rows, warnings = build_selection_rows(runs)

        self.assertEqual(warnings, [])
        self.assertEqual(rows[0]['chosen'], 'APCM-CE-FD5L1')
        self.assertEqual(rows[0]['selection_seeds'], '2024')
        self.assertEqual(rows[0]['report_seeds'], '2024,2025,2026')
        self.assertAlmostEqual(rows[0]['mean_vali_mse'], 0.20)
        self.assertAlmostEqual(rows[0]['mse'], 0.40)

    def test_skips_when_required_seed_is_missing(self):
        runs = [
            run('GTR-FD5L1', 2024, 0.30, 0.30),
            run('APCM-FD5L1', 2024, 0.25, 0.25),
            run('APCM-CE-FD5L1', 2024, 0.20, 0.20),
            run('APCM-CE-FD5L1', 2025, 0.19, 0.19),
        ]

        rows, warnings = build_selection_rows(runs, selection_seeds=[2024, 2025])

        self.assertEqual(rows, [])
        self.assertEqual(len(warnings), 1)

    def test_runner_records_same_selected_checkpoint_metrics(self):
        output = (
            'Epoch: 1, Steps: 2 | Train Loss: 1.0 Vali Loss: 0.8 '
            'Vali MSE: 0.7 Vali MAE: 0.6\n'
            'Best checkpoint | Vali Loss: 0.8123 Vali MSE: 0.4567 Vali MAE: 0.3456\n'
            'mse:0.2345, mae:0.1234\n'
        )

        parsed = parse_run_output(output)

        self.assertEqual(parsed['best_vali_loss'], 0.8123)
        self.assertEqual(parsed['best_vali_mse'], 0.4567)
        self.assertEqual(parsed['mse'], 0.2345)

    def test_runner_rejects_legacy_log_without_selected_val_mse(self):
        output = 'Vali Loss: 0.8 Test Loss: 0.7\nmse:0.2, mae:0.1\n'
        self.assertIsNone(parse_run_output(output))


if __name__ == '__main__':
    unittest.main()
