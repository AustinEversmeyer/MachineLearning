#include <cmath>
#include <stdexcept>
#include "naive_bayes/gaussian.h"
#include "naive_bayes/types.h"

namespace naive_bayes {

Gaussian::Gaussian(double mean, double sigma)
    : mean_(mean), sigma_(sigma) {
  if (!(sigma_ > 0.0)) {
    throw std::invalid_argument("sigma must be > 0");
  }
  double var = sigma_ * sigma_;
  log_norm_ = -0.5 * std::log(kTwoPi * var);
  inv2var_ = 1.0 / (2.0 * var);
}

double Gaussian::LogPdf(double x) const {
  double d = x - mean_;
  return log_norm_ - d * d * inv2var_;
}

double Gaussian::Sample(std::mt19937& rng) const {
  std::normal_distribution<double> dist(mean_, sigma_);
  return dist(rng);
}

std::string Gaussian::TypeName() const {
  return "gaussian";
}

std::vector<double> Gaussian::Params() const {
  return {mean_, sigma_};
}

}  // namespace naive_bayes
