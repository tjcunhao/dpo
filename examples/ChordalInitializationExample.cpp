
/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_types.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/QuadraticProblem.h>

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>

using namespace std;
using namespace DPGO;

void chordal(string data_file)
{
  size_t n;
  data_file = "../../data/" + data_file + ".g2o";
  vector<RelativeSEMeasurement> dataset = read_g2o_file(data_file, n);
  size_t d = (!dataset.empty() ? dataset[0].t.size() : 0);
  cout << "Loaded dataset from file " << data_file << "." << endl;

  // Construct optimization problem
  SparseMatrix QCentral = constructConnectionLaplacianSE(dataset);
  QuadraticProblem problemCentral(n, d, d);
  problemCentral.setQ(QCentral);

  // Compute chordal relaxation
  Matrix TChordal = chordalInitialization(d, n, dataset);
  assert((unsigned)TChordal.rows() == d);
  assert((unsigned)TChordal.cols() == (d + 1) * n);
  std::cout << "Chordal initialization cost: " << 2 * problemCentral.f(TChordal) << std::endl;
  std::cout << "Chordal initialization grad: " << problemCentral.RieGrad(TChordal).norm() << std::endl;
}

int main(int argc, char **argv)
{
  /**
  ###########################################
  Parse input dataset
  ###########################################
  */

  vector<string> data_files{
      // "tinyGrid3D",
      "kitti_07",
      "sphere2500",
      "smallGrid3D",
      "kitti_02",
      "cubicle",
      "input_INTEL_g2o",
      "torus3D",
      "kitti_09",
      "city10000",
      "parking-garage",
      "kitti_06",
      "kitti_05",
      "ais2klinik",
      "kitti_00",
      "rim",
      "CSAIL",
      "kitti_08",
      "grid3D",
      "sphere_bignoise_vertex3",
      "input_MITb_g2o",
      "input_M3500_g2o"};
  for (string &data_file : data_files)
  {
    chordal(data_file);
  }
  exit(0);
}
