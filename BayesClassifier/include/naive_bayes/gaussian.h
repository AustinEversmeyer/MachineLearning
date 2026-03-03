#pragma once
#include "naive_bayes/distribution.h"

namespace naive_bayes {

class Gaussian : public FeatureDistribution {
 public:
  Gaussian(double mean, double sigma);
  double LogPdf(double x) const override;
  double Sample(std::mt19937& rng) const override;
  std::string TypeName() const override;
  std::vector<double> Params() const override;

 private:
  double mean_;
  double sigma_;
  double log_norm_;
  double inv2var_;
};

}  // namespace naive_bayes
