#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <stdlib.h>
#include <thread>
#include <unordered_map>
#include <vector>
#include <iomanip>

#include "recommender.h"


Dataset::Dataset(const std::string &filename) {
    max_user_ = -1;
    max_item_ = -1;
    num_tuples_ = 0;
    std::ifstream infile(filename);
    std::string line;

    // Discard header.
    assert(std::getline(infile, line));

    // Read the data.
    while (std::getline(infile, line)) {
        int pos = line.find(',');
        int user = std::atoi(line.substr(0, pos).c_str());
        int item = std::atoi(line.substr(pos + 1).c_str());
        by_user_[user].push_back({item, num_tuples_});
        by_item_[item].push_back({user, num_tuples_});
        max_user_ = std::max(max_user_, user);
        max_item_ = std::max(max_item_, item);
        ++num_tuples_;
    }
    std::cout << "max_user=" << max_user()
              << "\tmax_item=" << max_item()
              << "\tdistinct user=" << by_user_.size()
              << "\tdistinct item=" << by_item_.size()
              << "\tnum_tuples=" << num_tuples()
              << std::endl;
}


VectorXf Recommender::EvaluateUser(
        const VectorXf &all_scores,
        const SpVector &ground_truth,
        const SpVector &exclude) {
    VectorXf scores = all_scores;
    for (int i = 0; i < exclude.size(); ++i) {
        assert(exclude[i].first < scores.size());
        scores[exclude[i].first] = std::numeric_limits<float>::lowest();
    }

    std::vector<size_t> topk(scores.size());
    std::iota(topk.begin(), topk.end(), 0);
    std::stable_sort(topk.begin(), topk.end(),
                     [&scores](size_t i1, size_t i2) {
                         return scores[i1] > scores[i2];
                     });
    auto recall = [](int k, const std::set<int> &gt_set,
                     const std::vector<size_t> &topk) -> float {
        double result = 0.0;
        for (int i = 0; i < k; ++i) {
            if (gt_set.find(topk[i]) != gt_set.end()) {
                result += 1.0;
            }
        }
        return result / std::min<float>(k, gt_set.size());
    };

    auto ndcg = [](int k, const std::set<int> &gt_set,
                   const std::vector<size_t> &topk) -> float {
        double result = 0.0;
        for (int i = 0; i < k; ++i) {
            if (gt_set.find(topk[i]) != gt_set.end()) {
                result += 1.0 / log2(i + 2);
            }
        }
        double norm = 0.0;
        for (int i = 0; i < std::min<int>(k, gt_set.size()); ++i) {
            norm += 1.0 / log2(i + 2);
        }
        return result / norm;
    };

    std::set<int> gt_set;
    std::transform(ground_truth.begin(), ground_truth.end(),
                   std::inserter(gt_set, gt_set.begin()),
                   [](const std::pair<int, int> &p) { return p.first; });
    VectorXf result(3);
    result << recall(20, gt_set, topk),
            recall(50, gt_set, topk),
            ndcg(100, gt_set, topk);
    return result;
}

VectorXf Recommender::EvaluateDataset(
        const Dataset &data, const SpMatrix &eval_by_user) {
    return EvaluateDatasetInternal(
            data, eval_by_user,
            [&](const int user_id, const SpVector &history) -> VectorXf {
                return Score(user_id, history);
            });
}

void Recommender::writeOutput(const std::string &results_file,
                              const std::string &algorithm_name,
                              const std::unordered_map<std::string, std::string> &flags,
                              const VectorXf &metrics,
                              const std::chrono::milliseconds &train_time) {
    std::string dataset = flags.at("train_data");
    uint ind2 = dataset.rfind('/');
    uint ind1 = dataset.rfind('/', ind2 - 1) + 1;
    dataset = dataset.substr(ind1, ind2 - ind1);

    std::ofstream outfile(results_file);

    // Header
    outfile << "jobname,dataset,algorithm,";
    // params
    outfile << "embedding_dim,unobserved_weight,regularization,regularization_exp,stddev,epochs,block_size,";
    // results
    outfile << "Rec20,Rec50,NDCG100,Runtime" << std::endl;

    // Data & Algo
    outfile << results_file.substr(0, results_file.length() - 4) << ",";
    outfile << dataset << ",";
    outfile << algorithm_name << ",";
    // parameters
    outfile << flags.at("embedding_dim") << ",";
    outfile << flags.at("unobserved_weight") << ",";
    outfile << flags.at("regularization") << ",";
    outfile << flags.at("regularization_exp") << ",";
    outfile << flags.at("stddev") << ",";
    outfile << flags.at("epochs") << ",";
    outfile << flags.at("block_size") << ",";
    // results
    outfile << std::fixed << std::setprecision(4) << metrics[0] << ",";
    outfile << std::fixed << std::setprecision(4) << metrics[1] << ",";
    outfile << std::fixed << std::setprecision(4) << metrics[2] << ",";
    outfile << std::to_string(train_time.count()) << std::endl;

    outfile.close();
}
