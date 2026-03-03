#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "naive_bayes/rayleigh.h"

namespace naive_bayes {

Rayleigh::Rayleigh(double sigma)
    : sigma_(sigma) {
  if (!(sigma_ > 0.0)) {
    throw std::invalid_argument("sigma must be > 0");
  }
  log_sigma2_ = 2.0 * std::log(sigma_);
  double s2 = sigma_ * sigma_;
  inv2sigma2_ = 1.0 / (2.0 * s2);
}

double Rayleigh::LogPdf(double x) const {
  if (x <= 0.0) {
    return -std::numeric_limits<double>::infinity();
  }
  return std::log(x) - log_sigma2_ - x * x * inv2sigma2_;
}

double Rayleigh::Sample(std::mt19937& rng) const {
  const double u = std::generate_canonical<double, 53>(rng);
  const double safe = std::max(1e-12, 1.0 - u);
  return sigma_ * std::sqrt(-2.0 * std::log(safe));
}

std::string Rayleigh::TypeName() const {
  return "rayleigh";
}

std::vector<double> Rayleigh::Params() const {
  return {sigma_};
}

}  // namespace naive_bayes
