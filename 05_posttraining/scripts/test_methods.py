"""Meaningful CPU invariants and a real NF4 CUDA smoke check."""
import json
import math
import unittest

import torch
from torch import nn

from data_pipeline import STAGE,prepare,tokenizer,tensors
from engine import dpo_loss,group_advantages,text_response
from peft import LoRALinear,quantize_nf4,quantization_metadata


class MethodsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((STAGE/'config.json').read_text(encoding='utf-8'))
        cls.data=prepare(cls.config)
        cls.tok=tokenizer(cls.config)

    def test_disjoint_splits_and_reward(self):
        sets=[{r['id'] for r in rows} for rows in self.data.values()]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(len(set.union(*sets)),1024)
        for rows in self.data.values():
            for r in rows:
                self.assertEqual(int(r['chosen']),int(r['a']>r['b']))

    def test_response_mask_excludes_prompt_and_padding(self):
        rows=self.data['train'][:5]
        x,y,mask=tensors(rows,self.tok,'cpu')
        for i,row in enumerate(rows):
            prompt=self.tok.encode(row['prompt'],add_bos=True)
            response=self.tok.encode(row['chosen'],add_eos=True)
            self.assertFalse(mask[i,:len(prompt)-1].any())
            self.assertEqual(y[i][mask[i]].tolist(),response)
            self.assertEqual(int(mask[i].sum()),len(response))

    def test_dpo_reference_offset_and_gradient(self):
        c=torch.tensor([3.],requires_grad=True)
        r=torch.tensor([1.],requires_grad=True)
        loss=dpo_loss(c,r,c.detach(),r.detach(),.1)
        self.assertAlmostEqual(float(loss.detach()),math.log(2),places=6)
        loss.backward()
        self.assertLess(float(c.grad),0)
        self.assertGreater(float(r.grad),0)

    def test_group_relative_credit(self):
        adv=group_advantages(torch.tensor([0.,0.,1.,1.,0.,0.,0.,0.]),4)
        self.assertTrue(torch.isfinite(adv).all())
        self.assertAlmostEqual(float(adv[:4].mean()),0)
        self.assertTrue((adv[4:]==0).all())

    def test_unknown_sample_is_zero_reward_not_crash(self):
        response=text_response(self.tok,[self.tok.unk_id])
        self.assertNotIn(response,['0','1'])

    def test_lora_starts_identity_and_receives_gradient(self):
        base=nn.Linear(16,12).requires_grad_(False)
        layer=LoRALinear(base,4)
        x=torch.randn(2,16)
        torch.testing.assert_close(layer(x),base(x))
        layer(x).square().sum().backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(float(layer.b.grad.norm()),0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_genuine_nf4_storage_and_input_gradient(self):
        model=nn.Sequential(nn.Linear(128,128,bias=False)).to(torch.bfloat16)
        model=quantize_nf4(model)
        meta=quantization_metadata(model)
        self.assertEqual(meta['packed_uint8_modules'],1)
        self.assertEqual(meta['nested_quant_modules'],1)
        x=torch.randn(2,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            model(x).float().square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__=='__main__':
    unittest.main(verbosity=2)
