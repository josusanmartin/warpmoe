#!/usr/bin/env python3
import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"scripts"))
import roofline
class RooflineTest(unittest.TestCase):
 def test_qwen_batch1(self):
  self.assertEqual(roofline.per_expert_bytes(2048,768),2359296)
  self.assertEqual(roofline.expected_distinct(128,8,1),8)
  self.assertAlmostEqual(roofline.weight_floor_us(8*2359296,936),20.164,places=2)
 def test_batch32(self):
  self.assertAlmostEqual(roofline.expected_distinct(128,8,32),111.77,delta=0.3)
 def test_reject_bad_bandwidth(self):
  with self.assertRaises(ValueError):roofline.weight_floor_us(1234,0)
if __name__=="__main__":unittest.main()
