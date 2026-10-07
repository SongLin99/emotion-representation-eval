import unittest
import importlib.util
from evaluation.schema import invariance_rows, readout_rows
from evaluation.utils import EMOTIONS, fractional_top1, spearman
from evaluation.model_utils import pool_hidden_states
from evaluation.judge_responses import anonymous_payloads
from evaluation.robustness import response_quality,event_calibration,LEGACY_QUALITY_PROTOCOL


class CorrectionsTests(unittest.TestCase):
    def test_slope_maxima_tied_not_forced_winner(self):
        for i in range(6):
            self.assertAlmostEqual(fractional_top1([-.6]*6,i)[0],1/6)
        self.assertEqual(fractional_top1([1,1,.5],2),(0,2))

    def test_fresh_specificity_schema(self):
        r=dict(id='x',topic_id='t',language='en',emotion='fear',version='original',
               **{'cos_'+e:0.1 for e in EMOTIONS})
        self.assertEqual(readout_rows([r],'specificity')[0]['condition'],'fear')

    def test_fresh_sensitivity_schema(self):
        r=dict(id='x',topic_id='t',language='en',emotion='fear',level=0,
               **{'cos_'+e:0.1 for e in EMOTIONS})
        self.assertEqual(readout_rows([r],'sensitivity')[0]['mean_fear'],.1)

    def test_invariance_pairing_and_missing_rejected(self):
        rows=[dict(id='x',topic_id='t',emotion='fear',level=0,language=l,version=v,cos_fear=.2)
              for l in ('en','zh') for v in ('original','paraphrase')]
        self.assertEqual(invariance_rows(rows)[0]['paraphrase_zh_target'],.2)
        with self.assertRaises(ValueError):invariance_rows(rows[:-1])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):spearman([0,1],[float('nan'),1])

    def test_mixed_protocol_rejected(self):
        rows=[dict(id=str(l),topic_id='t',language='en',emotion='fear',layer=l,
                   **{'cos_'+e:0 for e in EMOTIONS}) for l in (16,24)]
        with self.assertRaises(ValueError):readout_rows(rows,'specificity')

    def test_anonymous_payload_has_no_conditions(self):
        data=[dict(id='fear_minus',emotion='fear',alpha=-.5,prompt='P',response='R')]
        self.assertEqual(set(anonymous_payloads(data)[0][1]),{'id','prompt','response'})

    def test_objective_quality_overrides_generous_judge(self):
        row=dict(response='这是损坏的�文本',prompt_language='zh',hit_token_limit=False,
                 judge={'coherence':5,'task_adherence':5,'language_adherence':5})
        self.assertFalse(response_quality(row)['valid'])
        row['response']='这是一段正常的中文回答';self.assertTrue(response_quality(row)['valid'])

    def test_length_cap_alone_is_not_a_quality_failure(self):
        row=dict(response='I finish work, organize the papers, and prepare for tomorrow.',
            prompt_language='en',hit_token_limit=True,
            judge={'coherence':4,'task_adherence':4,'language_adherence':4})
        self.assertTrue(response_quality(row)['valid'])
        self.assertIn('token_limit',response_quality(row,LEGACY_QUALITY_PROTOCOL)['reasons'])
        row['judge']['coherence']=2
        self.assertFalse(response_quality(row)['valid'])

    def test_repeated_loop_rejected_even_when_judge_is_generous(self):
        for language,text in [('en','A moment of pure bliss, '*12),('zh','与每一个温柔的满足，'*15)]:
            row=dict(response=text,prompt_language=language,hit_token_limit=False,
                judge={'coherence':5,'task_adherence':5,'language_adherence':5})
            self.assertIn('excessive_repetition',response_quality(row)['reasons'])

    def test_long_varied_text_and_short_emphasis_are_allowed(self):
        text=' '.join(f'I sort document {i} into its folder.' for i in range(100))
        row=dict(response=text,prompt_language='en',hit_token_limit=True,
            judge={'coherence':5,'task_adherence':5,'language_adherence':5})
        self.assertTrue(response_quality(row)['valid'])
        row['response']='Yes, yes, yes. I will finish the work tomorrow.'
        self.assertTrue(response_quality(row)['valid'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch is not installed')
    def test_continuation_pooling_ignores_shared_prefix(self):
        import torch
        from evaluation.model_utils import pool_continuation
        h=torch.tensor([[[100.],[100.],[1.],[3.]]]);mask=torch.ones(1,4)
        spans=[[[0,1],[1,2],[2,3],[3,4]]]
        self.assertEqual(pool_continuation(h,mask,spans,[2]).item(),2.)

    def test_bootstrap_excludes_all_copies_of_original_event(self):
        import numpy as np
        neutral=np.array([[100.],[10.],[20.]])
        mu,sd=event_calibration(neutral,[2,1,1])
        self.assertEqual(mu[0,0],15.)
        self.assertAlmostEqual(sd[0,0],np.std([10,20],ddof=1))


if __name__=='__main__':unittest.main()
