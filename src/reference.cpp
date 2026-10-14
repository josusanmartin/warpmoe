#include "warpmoe/format.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace warpmoe {
namespace {
std::uint32_t next_random(std::uint32_t& state) {
 state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
}
std::size_t nibble_offset(const Shape& s,int expert,Projection p,int row,int col){
 if(expert<0||expert>=s.experts)throw std::out_of_range("expert");
 int rows=p==Projection::Down?s.hidden:s.intermediate;
 int cols=p==Projection::Down?s.intermediate:s.hidden;
 if(row<0||row>=rows||col<0||col>=cols)throw std::out_of_range("matrix index");
 std::size_t base=std::size_t(expert)*s.elements_per_expert();
 if(p==Projection::Up)base+=std::size_t(s.hidden)*s.intermediate;
 if(p==Projection::Down)base+=2ULL*s.hidden*s.intermediate;
 return base+std::size_t(row)*cols+col;
}
std::size_t scale_offset(const Shape& s,int expert,Projection p,int row,int col){
 int cols=p==Projection::Down?s.intermediate:s.hidden;
 std::size_t base=std::size_t(expert)*s.scales_per_expert();
 if(p==Projection::Up)base+=std::size_t(s.intermediate)*(s.hidden/s.group_size);
 if(p==Projection::Down)base+=2ULL*s.intermediate*(s.hidden/s.group_size);
 return base+std::size_t(row)*(cols/s.group_size)+col/s.group_size;
}
float silu(float a){return a/(1.0f+std::exp(-a));}
}
void set_quant(QuantWeights& w,int e,Projection p,int row,int col,int value){
 if(value< -8||value>7)throw std::invalid_argument("int4 must be within [-8,7]");
 auto i=nibble_offset(w.shape,e,p,row,col);auto& byte=w.packed.at(i/2);
 std::uint8_t n=static_cast<std::uint8_t>(value+8);
 if(i&1)byte=static_cast<std::uint8_t>((byte&0x0f)|(n<<4));
 else byte=static_cast<std::uint8_t>((byte&0xf0)|n);
}
float dequant(const QuantWeights& w,int e,Projection p,int row,int col){
 auto i=nibble_offset(w.shape,e,p,row,col);std::uint8_t byte=w.packed.at(i/2);
 int n=(i&1)?(byte>>4):(byte&15);
 return float(n-8)*w.scales.at(scale_offset(w.shape,e,p,row,col));
}
QuantWeights make_weights(const Shape& s,std::uint32_t seed){
 s.validate();QuantWeights w{s,std::vector<std::uint8_t>(s.all_expert_weight_bytes()),
 std::vector<float>(std::size_t(s.experts)*s.scales_per_expert())};
 for(auto& byte:w.packed)byte=static_cast<std::uint8_t>(next_random(seed)&255);
 for(auto& scale:w.scales)scale=(1+(next_random(seed)&3))*(1.0f/256.0f);
 return w;
}
Inputs make_inputs(const Shape& s,std::uint32_t seed){
 s.validate();Inputs in{s,std::vector<float>(std::size_t(s.batch)*s.hidden),
 std::vector<std::int32_t>(s.pairs()),std::vector<float>(s.pairs())};
 for(auto& x:in.x)x=(int(next_random(seed)%2001)-1000)/1000.0f;
 for(int b=0;b<s.batch;++b)for(int k=0;k<s.topk;++k){
 in.route[std::size_t(b)*s.topk+k]=(b*7+k)%s.experts;
 in.scores[std::size_t(b)*s.topk+k]=1.0f/s.topk;
 }
 return in;
}
std::vector<float> reference_forward(const QuantWeights& w,const Inputs& in){
 w.shape.validate();const Shape& s=w.shape;
 if(s.batch!=in.shape.batch||s.hidden!=in.shape.hidden||
 s.intermediate!=in.shape.intermediate||s.experts!=in.shape.experts||
 s.topk!=in.shape.topk||s.group_size!=in.shape.group_size||
 in.x.size()!=std::size_t(s.batch)*s.hidden||
 in.route.size()!=s.pairs()||in.scores.size()!=s.pairs()||
 w.packed.size()!=s.all_expert_weight_bytes()||
 w.scales.size()!=std::size_t(s.experts)*s.scales_per_expert())
 throw std::invalid_argument("shape and buffer mismatch");
 std::vector<float> result(std::size_t(s.batch)*s.hidden,0),hidden(s.intermediate);
 for(int b=0;b<s.batch;++b)for(int slot=0;slot<s.topk;++slot){
 int pair=b*s.topk+slot,e=in.route[pair];
 if(e<0||e>=s.experts)throw std::invalid_argument("invalid route");
 for(int r=0;r<s.intermediate;++r){
 float g=0,u=0;
 for(int c=0;c<s.hidden;++c){
 float x=in.x[std::size_t(b)*s.hidden+c];
 g+=x*dequant(w,e,Projection::Gate,r,c);
 u+=x*dequant(w,e,Projection::Up,r,c);
 }
 hidden[r]=silu(g)*u;
 }
 for(int r=0;r<s.hidden;++r){
 float y=0;
 for(int c=0;c<s.intermediate;++c)y+=hidden[c]*dequant(w,e,Projection::Down,r,c);
 result[std::size_t(b)*s.hidden+r]+=in.scores[pair]*y;
 }
 }
 return result;
}
std::size_t unique_expert_weight_bytes(const Shape& s,const std::vector<std::int32_t>& route){
 s.validate();if(route.size()!=s.pairs())throw std::invalid_argument("route length mismatch");
 std::vector<bool> seen(s.experts,false);std::size_t distinct=0;
 for(auto e:route){
 if(e<0||e>=s.experts)throw std::invalid_argument("route expert out of bounds");
 if(!seen[e]){seen[e]=true;++distinct;}
 }
 return distinct*s.bytes_per_expert();
}
double ideal_weight_us(std::size_t bytes,double bandwidth_gb_s){
 if(!(bandwidth_gb_s>0)||!std::isfinite(bandwidth_gb_s))
 throw std::invalid_argument("bandwidth must be finite and positive");
 return double(bytes)/(bandwidth_gb_s*1e9)*1e6;
}
} // namespace warpmoe
