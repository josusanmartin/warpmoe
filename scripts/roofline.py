#!/usr/bin/env python3
"""Analytical weight traffic floor; not a measured kernel-speed predictor."""
import argparse
import math
def expected_distinct(experts:int,topk:int,batch:int)->float:
    if not(0<topk<=experts and batch>=1):
        raise ValueError("invalid experts/topk/batch")
    return experts*(1-(1-topk/experts)**batch)
def per_expert_bytes(hidden:int,intermediate:int,bitwidth:int=4)->float:
    if hidden<=0 or intermediate<=0 or bitwidth<=0:
        raise ValueError("dimensions and bitwidth must be positive")
    return 3*hidden*intermediate*bitwidth/8
def weight_floor_us(bytes_read:float,bandwidth_gbs:float)->float:
    if not math.isfinite(bandwidth_gbs) or bandwidth_gbs<=0:
        raise ValueError("bandwidth must be positive")
    return bytes_read/(bandwidth_gbs*1e9)*1e6
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--experts",type=int,default=128)
    p.add_argument("--topk",type=int,default=8)
    p.add_argument("--hidden",type=int,default=2048)
    p.add_argument("--intermediate",type=int,default=768)
    p.add_argument("--bits",type=int,default=4)
    p.add_argument("--bandwidth-gbs",type=float,default=936.)
    p.add_argument("--baseline-us",type=float)
    a=p.parse_args()
    distinct=expected_distinct(a.experts,a.topk,a.batch)
    size=per_expert_bytes(a.hidden,a.intermediate,a.bits)
    floor=weight_floor_us(size*distinct,a.bandwidth_gbs)
    print(f"Expected distinct experts under IID uniform routing: {distinct:.2f} / {a.experts}")
    print(f"Expert packed bytes: {size/1024**2:.3f} MiB")
    print(f"Total packed weight bytes: {size*distinct/1024**2:.3f} MiB")
    print(f"Ideal streaming floor at {a.bandwidth_gbs:.1f} GB/s: {floor:.3f} us")
    print("Excludes scales, duplicate reads, activations, routing, and cache effects.")
    if a.baseline_us is not None:
        if a.baseline_us<=0:p.error("--baseline-us must be positive")
        print(f"Estimated floor / baseline: {floor/a.baseline_us:.1%}")
        print(f"Optimistic headroom (baseline / floor): {a.baseline_us/floor:.2f}x")
if __name__=="__main__":main()
