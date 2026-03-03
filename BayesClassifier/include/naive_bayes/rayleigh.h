#pragma once
#include "naive_bayes/distribution.h"

namespace naive_bayes {

class Rayleigh : public FeatureDistribution {
 public:
  explicit Rayleigh(double sigma);
  double LogPdf(double x) const override;
  double Sample(std::mt19937& rng) const override;
  std::string TypeName() const override;
  std::vector<double> Params() const override;

 private:
  double sigma_;
  double log_sigma2_;
  double inv2sigma2_;
};

}  // namespace naive_bayes
