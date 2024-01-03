
/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_types.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/PGOAgent.h>
#include <DPGO/QuadraticProblem.h>

#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;
using namespace DPGO;

void compute(size_t num_robots, string file_name, bool is_partition, string strongth = "strong")
{
    cout << "Multi-robot pose graph optimization example. " << endl;
    if (num_robots <= 0)
    {
        cout << "Number of robots must be positive!" << endl;
        exit(1);
    }
    cout << "Simulating " << num_robots << " robots." << endl;

    size_t num_poses;
    string data_file_path = "../../data/" + file_name + ".g2o";
    vector<RelativeSEMeasurement> dataset = read_g2o_file(data_file_path, num_poses);
    cout << "Loaded dataset from file "
         << file_name
         << ".g2o." << endl;

    /**
    ###########################################
    Options
    ###########################################
    */
    unsigned int n, d, r;
    d = (!dataset.empty() ? dataset[0].t.size() : 0);
    n = num_poses;
    r = 5;
    bool acceleration = false;
    bool verbose = false;
    unsigned numIters = 1000;

    // Construct the centralized problem (used for evaluation)
    SparseMatrix QCentral = constructConnectionLaplacianSE(dataset);
    QuadraticProblem problemCentral(n, d, r);
    problemCentral.setQ(QCentral);

    /**
    ###########################################
    Partition dataset into robots
    ###########################################
    */
    unsigned int num_poses_per_robot = num_poses / num_robots;
    if (num_poses_per_robot <= 0)
    {
        cout << "More robots than total number of poses! Decrease the number of robots" << endl;
        exit(1);
    }

    // create mapping from global pose index to local pose index
    map<unsigned, PoseID> PoseMap;
    // create mapping from local pose index to global pose index
    map<PoseID, unsigned> Pose2Index;
    vector<unsigned> pose_counts(num_robots, 0);
    bool is_graph_partition = is_partition;
    if (is_graph_partition)
    {
        string partition_file = "../../graph/" + to_string(num_robots) + "/" + strongth + "/" + file_name;
        ifstream f(partition_file);
        size_t cur_num = 0;
        string cur_group;

        while (getline(f, cur_group))
        {
            int group = stoi(cur_group);
            PoseID pose = make_pair(group, pose_counts[group]);
            PoseMap[cur_num] = pose;
            Pose2Index[pose] = cur_num;
            ++cur_num;
            ++pose_counts[group];
        }
    }
    else
    {
        for (unsigned robot = 0; robot < (unsigned)num_robots; ++robot)
        {
            unsigned startIdx = robot * num_poses_per_robot;
            unsigned endIdx = (robot + 1) * num_poses_per_robot; // non-inclusive
            if (robot == (unsigned)num_robots - 1)
                endIdx = n;
            pose_counts[robot] = endIdx - startIdx;
            for (unsigned idx = startIdx; idx < endIdx; ++idx)
            {
                unsigned localIdx = idx - startIdx; // this is the local ID of this pose
                PoseID pose = make_pair(robot, localIdx);
                PoseMap[idx] = pose;
                Pose2Index[pose] = idx;
            }
        }
    }

    vector<vector<RelativeSEMeasurement>> odometry(num_robots);
    vector<vector<RelativeSEMeasurement>> private_loop_closures(num_robots);
    vector<vector<RelativeSEMeasurement>> shared_loop_closure(num_robots);
    for (auto mIn : dataset)
    {
        PoseID src = PoseMap[mIn.p1];
        PoseID dst = PoseMap[mIn.p2];

        unsigned srcRobot = src.first;
        unsigned srcIdx = src.second;
        unsigned dstRobot = dst.first;
        unsigned dstIdx = dst.second;

        RelativeSEMeasurement m(srcRobot, dstRobot, srcIdx, dstIdx, mIn.R, mIn.t,
                                mIn.kappa, mIn.tau);

        if (srcRobot == dstRobot)
        {
            // private measurement
            if (mIn.p1 + 1 == mIn.p2)
            {
                // Odometry
                odometry[srcRobot].push_back(m);
            }
            else
            {
                // private loop closure
                private_loop_closures[srcRobot].push_back(m);
            }
        }
        else
        {
            // shared measurement
            shared_loop_closure[srcRobot].push_back(m);
            shared_loop_closure[dstRobot].push_back(m);
        }
    }

    /**
    ###########################################
    Initialization
    ###########################################
    */
    vector<PGOAgent *> agents;
    for (unsigned robot = 0; robot < (unsigned)num_robots; ++robot)
    {
        PGOAgentParameters options(d, r, num_robots);
        options.acceleration = acceleration;
        options.verbose = verbose;

        auto *agent = new PGOAgent(robot, options);

        // All agents share a special, common matrix called the 'lifting matrix' which the first agent will generate
        if (robot > 0)
        {
            Matrix M;
            agents[0]->getLiftingMatrix(M);
            agent->setLiftingMatrix(M);
        }

        agent->setPoseGraph(odometry[robot], private_loop_closures[robot],
                            shared_loop_closure[robot]);
        agents.push_back(agent);
    }

    /**
    ##########################################################################################
    For this demo, we initialize each robot's estimate from the centralized chordal relaxation
    ##########################################################################################
    */
    Matrix TChordal = chordalInitialization(d, n, dataset);
    Matrix XChordal = fixedStiefelVariable(d, r) * TChordal; // Lift estimate to the correct relaxation rank
    for (unsigned robot = 0; robot < (unsigned)num_robots; ++robot)
    {

        Matrix X(r, pose_counts[robot] * (d + 1));
        for (size_t i = 0; i < pose_counts[robot]; ++i)
        {
            X.block(0, i * (d + 1), r, d + 1) = XChordal.block(0, Pose2Index[make_pair(robot, i)] * (d + 1), r, d + 1);
        }
        // cout << robot << " Xinit :" << endl;
        // cout << X << endl;
        // cout << "*****************************************************" << endl;
        // cout << XChordal.block(0, startIdx * (d + 1), r, (endIdx - startIdx) * (d + 1)) << endl;
        // cout << endl;
        agents[robot]->setX(X);
    }

    /**
    ###########################################
    Optimization loop
    ###########################################
    */
    Matrix Xopt(r, n * (d + 1));
    unsigned selectedRobot = 0;
    cout << "Running " << numIters << " iterations..." << endl;

    std::fstream file;
    string filename;
    // filename += "../../result/graph/";
    // if (!acceleration)
    // {
    //   filename += "NA";
    // }
    if (is_graph_partition)
        filename = strongth + file_name;
    else
        filename = "../../result/opt_pose/NP" + file_name;
    filename += ".txt";
    file.open(filename, std::fstream::trunc | std::fstream::out);

    for (unsigned iter = 0; iter < numIters; ++iter)
    {
        PGOAgent *selectedRobotPtr = agents[selectedRobot];

        // Non-selected robots perform an iteration
        for (auto *robotPtr : agents)
        {
            assert(robotPtr->instance_number() == 0);
            assert(robotPtr->iteration_number() == iter);
            if (robotPtr->getID() != selectedRobot)
            {
                robotPtr->iterate(false);
            }
        }

        // Selected robot requests public poses from others
        for (auto *robotPtr : agents)
        {
            if (robotPtr->getID() == selectedRobot)
                continue;
            PoseDict sharedPoses;
            if (!robotPtr->getSharedPoseDict(sharedPoses))
            {
                continue;
            }
            selectedRobotPtr->setNeighborStatus(robotPtr->getStatus());
            selectedRobotPtr->updateNeighborPoses(robotPtr->getID(), sharedPoses);
        }

        // When using acceleration, selected robot also requests auxiliary poses
        if (acceleration)
        {
            for (auto *robotPtr : agents)
            {
                if (robotPtr->getID() == selectedRobot)
                    continue;
                PoseDict auxSharedPoses;
                if (!robotPtr->getAuxSharedPoseDict(auxSharedPoses))
                {
                    continue;
                }
                selectedRobotPtr->setNeighborStatus(robotPtr->getStatus());
                selectedRobotPtr->updateAuxNeighborPoses(robotPtr->getID(), auxSharedPoses);
            }
        }

        // Selected robot update
        selectedRobotPtr->iterate(true);

        // Form centralized solution
        for (unsigned robot = 0; robot < (unsigned)num_robots; ++robot)
        {

            Matrix XRobot;
            if (agents[robot]->getX(XRobot))
            {
                for (size_t i = 0; i < pose_counts[robot]; ++i)
                {
                    Xopt.block(0, Pose2Index[make_pair(robot, i)] * (d + 1), r, (d + 1)) = XRobot.block(0, i * (d + 1), r, d + 1);
                }
            }
        }
        Matrix RGrad = problemCentral.RieGrad(Xopt);
        double RGradNorm = RGrad.norm();
        double result = 2 * problemCentral.f(Xopt);
        std::cout << std::setprecision(5)
                  << "Iter = " << iter << " | "
                  << "robot = " << selectedRobotPtr->getID() << " | "
                  << "cost = " << result << " | "
                  << "gradnorm = " << RGradNorm << std::endl;

        // // Exit if gradient norm is sufficiently small
        // if (RGradNorm < 0.1)
        // {
        //   break;
        // }

        // Select next robot with largest gradient norm
        std::vector<unsigned> neighbors = selectedRobotPtr->getNeighbors();
        double selected_max_norm = 0;
        if (neighbors.empty())
        {
            selectedRobot = selectedRobotPtr->getID();
        }
        else
        {
            std::vector<double> gradNorms;
            for (size_t robot = 0; robot < (unsigned)num_robots; ++robot)
            {

                Matrix RGradRobot(r, pose_counts[robot] * (d + 1));
                for (size_t i = 0; i < pose_counts[robot]; ++i)
                    RGradRobot.block(0, i * (d + 1), r, d + 1) = RGrad.block(0, Pose2Index[make_pair(robot, i)] * (d + 1), r, d + 1);
                gradNorms.push_back(RGradRobot.norm());
            }
            selectedRobot = std::max_element(gradNorms.begin(), gradNorms.end()) - gradNorms.begin();
            selected_max_norm = *std::max_element(gradNorms.begin(), gradNorms.end());
        }
        file << setprecision(10)
             << result << "," << RGradNorm << "," << selected_max_norm << std::endl;
        // Share global anchor for rounding
        Matrix M;
        agents[0]->getSharedPose(0, M);
        for (auto agentPtr : agents)
        {
            agentPtr->setGlobalAnchor(M);
        }
    }
    string output_file;

    if (is_graph_partition)
        output_file = "../../result/opt_pose/" + strongth + file_name + ".csv";
    else
        output_file = "../../result/opt_pose/NP" + file_name + ".csv";
    writeMatrixToFile(Xopt.block(0, 0, r, d).transpose() * Xopt, output_file);

    for (auto agentPtr : agents)
    {
        agentPtr->reset();
    }
}

int main(int argc, char **argv)
{
    // vector<string> data_files{"rim"};
    vector<string> data_files{
        // "tinyGrid3D",
        // "kitti_07",
        // "sphere2500",
        // "smallGrid3D",
        // "kitti_02",
        // "cubicle",
        // "input_INTEL_g2o",
        // "torus3D",
        // "kitti_09",
        // "city10000",
        "parking-garage",
        // "kitti_06",
        // "kitti_05",
        "ais2klinik",
        // "kitti_00",
        "rim",
        // "CSAIL",
        // "kitti_08",
        "grid3D",
        // "sphere_bignoise_vertex3",
        // "input_MITb_g2o",
        // "input_M3500_g2o"
    };
    vector<string> strongths{"fast", "eco", "strong", "highest"};
    for (string &file_name : data_files)
    {
        // for (string &strongth : strongths)
        // {
        //     compute(5, file_name, true, strongth);
        // }
        compute(5, file_name, false);
    }
    exit(0);
}
