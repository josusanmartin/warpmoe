#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace warpmoe {
enum class Projection : int { Gate = 0, Up = 1, Down = 2 };
struct Shape {
  int batch = 1, hidden = 32, intermediate = 32, experts = 8, topk = 2, group_size = 16;
  void validate() const {
    if (batch <= 0 || hidden <= 0 || intermediate <= 0 || experts <= 0 ||
        topk <= 0 || topk > experts || group_size <= 0 ||
        hidden % group_size || intermediate % group_size || hidden % 2 || intermediate % 2)
      throw std::invalid_argument("invalid shape / group alignment");
  }
  std::size_t elements_per_expert() const { return 3ULL * hidden * intermediate; }
  std::size_t bytes_per_expert() const { return elements_per_expert() / 2; }
  std::size_t scales_per_expert() const {
    return 2ULL * intermediate * (hidden / group_size) +
           1ULL * hidden * (intermediate / group_size);
  }
  std::size_t pairs() const { return std::size_t(batch) * topk; }
  std::size_t all_expert_weight_bytes() const { return std::size_t(experts) * bytes_per_expert(); }
};
struct QuantWeights { Shape shape; std::vector<std::uint8_t> packed; std::vector<float> scales; };
struct Inputs {
  Shape shape;
  std::vector<float> x;
  std::vector<std::int32_t> route;
  std::vector<float> scores;
};
struct Task { bool down; int pair, tile; };
struct TicketPlan {
  int gate_rows = 64, down_rows = 128;
  int gate_tiles(const Shape& s) const { return (s.intermediate + gate_rows - 1) / gate_rows; }
  int down_tiles(const Shape& s) const { return (s.hidden + down_rows - 1) / down_rows; }
  std::size_t total(const Shape& s) const {
    return s.pairs() * std::size_t(gate_tiles(s) + down_tiles(s));
  }
  Task at(const Shape& s, std::size_t ticket) const {
    if (ticket >= total(s)) throw std::out_of_range("ticket outside task plan");
    auto gate_total = s.pairs() * std::size_t(gate_tiles(s));
    if (ticket < gate_total) return {false, int(ticket / gate_tiles(s)), int(ticket % gate_tiles(s))};
    ticket -= gate_total;
    return {true, int(ticket / down_tiles(s)), int(ticket % down_tiles(s))};
  }
};
float dequant(const QuantWeights&, int expert, Projection, int row, int col);
void set_quant(QuantWeights&, int expert, Projection, int row, int col, int signed_value);
QuantWeights make_weights(const Shape&, std::uint32_t seed = 17);
Inputs make_inputs(const Shape&, std::uint32_t seed = 42);
std::vector<float> reference_forward(const QuantWeights&, const Inputs&);
std::size_t unique_expert_weight_bytes(const Shape&, const std::vector<std::int32_t>&);
double ideal_weight_us(std::size_t bytes, double bandwidth_gb_s);
} // namespace warpmoe
