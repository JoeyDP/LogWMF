// Copyright 2023 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LOGMF_EXPERIMENTS_RECOMMENDER_H
#define LOGMF_EXPERIMENTS_RECOMMENDER_H

#include <thread>


#include "Eigen/Dense"
#include "Eigen/Core"

using SpVector = std::vector<std::pair<int, int>>;
using SpMatrix = std::unordered_map<int, SpVector>;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        MatrixXf;
typedef Eigen::VectorXf VectorXf;


class Dataset {
public:
    explicit Dataset(const std::string &filename);

    const SpMatrix &by_user() const { return by_user_; }

    const SpMatrix &by_item() const { return by_item_; }

    const int max_user() const { return max_user_; }

    const int max_item() const { return max_item_; }

    const int num_tuples() const { return num_tuples_; }

private:
    SpMatrix by_user_;
    SpMatrix by_item_;
    int max_user_;
    int max_item_;
    int num_tuples_;
};


class Recommender {
public:

    virtual ~Recommender() {}

    virtual VectorXf Score(const int user_id, const SpVector &user_history) {
        return VectorXf::Zero(1);
    }

    virtual void Train(const Dataset &dataset) {}

    VectorXf EvaluateUser(const VectorXf &all_scores,
                          const SpVector &ground_truth,
                          const SpVector &exclude);

    // Templated implementation for evaluating a dataset. Requires a function that
    // scores all items for a given user or history.
    template<typename F>
    VectorXf EvaluateDatasetInternal(
            const Dataset &data, const SpMatrix &eval_by_user,
            F score_user_and_history) {
        std::mutex m;
        auto eval_by_user_iter = eval_by_user.begin();  // protected by m
        VectorXf metrics = VectorXf::Zero(3);

        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(std::thread([&] {
                while (true) {
                    // Get a new user to work on.
                    m.lock();
                    if (eval_by_user_iter == eval_by_user.end()) {
                        m.unlock();
                        return;
                    }
                    int u = eval_by_user_iter->first;
                    SpVector ground_truth = eval_by_user_iter->second;
                    ++eval_by_user_iter;
                    m.unlock();

                    // Process the user.
                    const SpVector &user_history = data.by_user().at(u);
                    VectorXf scores = score_user_and_history(u, user_history);
                    VectorXf this_metrics = this->EvaluateUser(scores, ground_truth,
                                                               user_history);

                    // Update the metric.
                    m.lock();
                    metrics += this_metrics;
                    m.unlock();
                }
            }));
        }

        // Join all threads.
        for (int i = 0; i < threads.size(); ++i) {
            threads[i].join();
        }
        metrics /= eval_by_user.size();
        return metrics;
    }

    // Common implementation for evaluating a dataset. It uses the scoring
    // function of the class.
    virtual VectorXf EvaluateDataset(
            const Dataset &data, const SpMatrix &eval_by_user);


    static void writeOutput(const std::string& results_file,
                            const std::string& algorithm_name,
                            const std::unordered_map<std::string, std::string> &flags,
                            const VectorXf& metrics,
                            const std::chrono::milliseconds& train_time);
};

#endif //LOGMF_EXPERIMENTS_RECOMMENDER_H
