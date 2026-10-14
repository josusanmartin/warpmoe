#include "warpmoe/format.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace warpmoe;
bool close(float a,float b,float tol=1e-3f){return std::abs(a-b)<tol;}
void packing_test(){
 Shape s{1,16,16,2,1,8};auto w=make_weights(s);
 for(int v=-8;v<8;++v){
 set_quant(w,1,Projection::Up,5,v+8,v);
 float reconstructed=dequant(w,1,Projection::Up,5,v+8);
 auto si=std::size_t(s.scales_per_expert())+
 std::size_t(s.intermediate)*(s.hidden/s.group_size)+
 5*(s.hidden/s.group_size)+(v+8)/s.group_size;
 assert(close(reconstructed,v*w.scales.at(si)));
 }
}
void deterministic_math_test(){
 Shape s{2,16,16,4,2,8};auto w=make_weights(s);
 std::fill(w.packed.begin(),w.packed.end(),0x99);
 std::fill(w.scales.begin(),w.scales.end(),1.0f);
 auto in=make_inputs(s);
 std::fill(in.x.begin(),in.x.end(),1.0f);
 std::fill(in.scores.begin(),in.scores.end(),0.5f);
 auto out=reference_forward(w,in);
 float activation=16.0f/(1.0f+std::exp(-16.0f));
 float expected=16.0f*(activation*16.0f);
 for(float x:out)assert(close(x,expected,0.01f));
}
void dynamic_tasks_test(){
 Shape s{3,64,192,8,2,16};TicketPlan plan;
 assert(plan.gate_tiles(s)==3&&plan.down_tiles(s)==1);
 assert(plan.total(s)==s.pairs()*4);
 for(std::size_t t=0;t<plan.total(s);++t){
 auto x=plan.at(s,t);
 if(t<s.pairs()*3){assert(!x.down);assert(x.tile<3);}
 else{assert(x.down);assert(x.tile==0);}
 }
}
void shape_floor_test(){
 Shape s{1,2048,768,128,8,128};s.validate();
 assert(s.bytes_per_expert()==2359296);
 std::vector<std::int32_t> route{0,1,2,3,4,5,6,7};
 auto bytes=unique_expert_weight_bytes(s,route);
 assert(bytes==18ULL*1024*1024);
 assert(std::abs(ideal_weight_us(bytes,936.)-20.162)<0.01);
}
void random_reference_test(){
 Shape s{3,32,64,16,4,16};
 auto a=make_weights(s,9),b=make_weights(s,9);
 assert(a.packed==b.packed&&a.scales==b.scales);
 auto x=make_inputs(s,9);
 auto y=reference_forward(a,x);
 for(float v:y)assert(std::isfinite(v));
 assert(y.size()==std::size_t(s.batch*s.hidden));
}
int main(){
 packing_test();deterministic_math_test();dynamic_tasks_test();
 shape_floor_test();random_reference_test();
 std::cout<<"CPU tests passed: packing, model math, ticket plan, roofline, determinism\n";
}
