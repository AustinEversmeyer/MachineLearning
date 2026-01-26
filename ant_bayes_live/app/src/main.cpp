#include <iostream>

#include "ant_bayes_live.h"

int main() {
  try {
    AntBayesLive demo;
    std::cout << "Ant live inference demo (in-memory JSON)\n";
    demo.RunLoop(3);
  } catch (const std::exception& ex) {
    std::cerr << "Inference failed: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
