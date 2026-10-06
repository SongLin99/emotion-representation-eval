"""Qwen eligibility must preserve full ladders and bilingual paired conditions."""
import unittest
from evaluation.qwen_summary import usable_tables


class QwenCoverageTests(unittest.TestCase):
    def rows(self):
        return [dict(id=f'fear_T01_L{level}', topic_id='fear_T01', level=level,
            language=lang, version=version, task=task, pooling='token50', valid='True')
            for lang in ('en', 'zh') for version, task in (
                ('original', 'sensitivity'), ('paraphrase', 'invariance')) for level in range(5)]

    def test_short_original_removes_ladder_but_only_its_paired_id(self):
        rows = self.rows()
        next(r for r in rows if r['language']=='zh' and r['version']=='original' and r['level']==0)['valid']='False'
        _, se, inv, missing, complete = usable_tables(rows)
        self.assertEqual(len(se), 5)
        self.assertEqual({r['language'] for r in se}, {'en'})
        self.assertEqual(missing, {('zh', 'fear_T01')})
        self.assertEqual(len(complete), 4)
        self.assertEqual(len(inv), 16)
        self.assertEqual(len(rows), 20)

    def test_short_paraphrase_preserves_valid_sensitivity_ladder(self):
        rows = self.rows()
        next(r for r in rows if r['language']=='zh' and r['version']=='paraphrase' and r['level']==0)['valid']='False'
        _, se, inv, missing, complete = usable_tables(rows)
        self.assertEqual(len(se), 10)
        self.assertEqual(missing, set())
        self.assertEqual(len(complete), 4)
        self.assertEqual(len(inv), 16)

    def test_missing_language_condition_is_not_a_complete_pair(self):
        rows = [r for r in self.rows() if not (r['language']=='zh' and r['version']=='paraphrase')]
        _, se, inv, missing, complete = usable_tables(rows)
        self.assertEqual(len(se), 10)
        self.assertEqual(missing, set())
        self.assertEqual(complete, set())
        self.assertEqual(inv, [])
