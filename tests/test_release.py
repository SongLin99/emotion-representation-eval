import importlib.util
import math
import unittest
from pathlib import Path
from evaluation.utils import auc, ranks, spearman, ROOT, read_jsonl
from evaluation.functionality import paired_contrasts
from evaluation.judge_responses import anonymous_payloads
from evaluation.run_functionality import make_jobs
from scripts.summarize_results import verified_metrics


class ReleaseTests(unittest.TestCase):
    def test_all_main_metrics_match_archive(self):
        for name, (value, archived) in verified_metrics().items():
            with self.subTest(metric=name):
                self.assertTrue(math.isclose(value, archived, rel_tol=1e-7, abs_tol=1e-8))

    def test_rank_ties_and_auroc(self):
        self.assertEqual(ranks([1, 3, 3, 2]), [1, 3.5, 3.5, 2])
        self.assertEqual(auc([2], [1, 2]), .75)
        self.assertAlmostEqual(spearman([0, 1, 2], [5, 3, 1]), -1)

    def test_vectors_exist(self):
        for language in ('en', 'zh'):
            self.assertTrue((ROOT/'vectors/olmo3_7b_instruct'/language/'emotion_vectors.safetensors').is_file())

    def test_functionality_condition_counts(self):
        import json
        base = json.loads((ROOT/'data/functionality/functionality_protocol.json').read_text())
        cross = json.loads((ROOT/'data/functionality/crosslingual_protocol.json').read_text())
        self.assertEqual(sum(len(list(make_jobs(task))) for task in base['tasks']), 300)
        self.assertEqual(sum(len(list(make_jobs(task, True))) for task in cross['tasks']), 600)

    def test_judge_payload_does_not_expose_condition(self):
        rows = [dict(id='desk_target_anger_+0.5', alpha=.5, emotion='anger', prompt='P', response='R'),
                dict(id='desk_target_anger_-0.5', alpha=-.5, emotion='anger', prompt='P', response='S')]
        payloads = anonymous_payloads(rows)
        self.assertEqual(sorted(index for index, _ in payloads), [0, 1])
        for _, payload in payloads:
            self.assertEqual(set(payload), {'id', 'prompt', 'response'})
            self.assertTrue(payload['id'].startswith('item_'))
            self.assertNotIn('anger', payload['id'])

    def test_missing_paired_condition_rejected(self):
        rows = read_jsonl(ROOT/'results/functionality/judged_responses.jsonl')
        target = next(row for row in rows if row['alpha'])
        with self.assertRaises(ValueError):
            paired_contrasts([target])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch is not installed')
class InterventionTests(unittest.TestCase):
    def test_pool_50th_token_inclusive_with_left_padding(self):
        import torch
        from evaluation.model_utils import pool_hidden_states
        hidden = torch.arange(104).reshape(2, 52, 1).float()
        mask = torch.ones(2, 52, dtype=torch.long)
        mask[0, :2] = 0
        pooled = pool_hidden_states(hidden, mask)
        self.assertEqual(pooled[0].item(), 51)
        self.assertEqual(pooled[1].item(), 102)
        with self.assertRaises(ValueError):
            pool_hidden_states(hidden[:, :49], mask[:, :49])

    def test_only_last_position_edited_and_hook_removed(self):
        import torch
        from types import SimpleNamespace
        from evaluation.model_utils import assistant_intervention
        block = torch.nn.Identity()
        model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
        hidden = torch.zeros(2, 5, 3)
        delta = torch.ones(2, 3)
        with assistant_intervention(model, delta, layer=1):
            output = block(hidden)
            self.assertTrue(torch.equal(output[:, :-1], hidden[:, :-1]))
            self.assertTrue(torch.equal(output[:, -1], delta))
            self.assertTrue(torch.equal(block(hidden[:, :1]), delta.unsqueeze(1)))
        self.assertTrue(torch.equal(block(hidden), hidden))


if __name__ == '__main__':
    unittest.main()
