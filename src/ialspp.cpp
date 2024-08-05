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


#include <algorithm>
#include <chrono>
#include <fstream>
#include <mutex>
#include <random>
#include <set>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Core"

#include "ialspp.h"
#include "recommender.h"


const VectorXf ProjectBlock(
        const SpVector &user_history,
        const VectorXf &user_embedding,
        const VectorXf &local_user_embedding,
        const MatrixXf &local_item_embedding,
        const VectorXf &prediction,
        const MatrixXf &local_gramian,
        const MatrixXf &local_global_gramian,
        const float reg, const float unobserved_weight) {
    assert(user_history.size() > 0);
    int local_embedding_dim = local_item_embedding.cols();
    assert(local_embedding_dim > 0);

    VectorXf new_value(local_embedding_dim);

    Eigen::MatrixXf matrix = unobserved_weight * local_gramian;

//    printf("Unobserved weight: %f\n", unobserved_weight);

    for (int i = 0; i < local_embedding_dim; ++i) {
        matrix(i, i) += reg;
    }

    const int kMaxBatchSize = 128;
    auto matrix_symm = matrix.selfadjointView<Eigen::Lower>();
    Eigen::VectorXf rhs = Eigen::VectorXf::Zero(local_embedding_dim);
    const int batch_size = std::min(static_cast<int>(user_history.size()),
                                    kMaxBatchSize);
    int num_batched = 0;
    Eigen::MatrixXf pos_factor_batch(local_embedding_dim, batch_size);
    Eigen::MatrixXf neg_factor_batch(local_embedding_dim, batch_size);
    for (const auto &item_and_rating_index: user_history) {
        const int cp = item_and_rating_index.first;
        const int rating_index = item_and_rating_index.second;
        assert(cp < local_item_embedding.rows());
        assert(rating_index < prediction.size());
        const VectorXf cp_v = local_item_embedding.row(cp);

        float score = prediction.coeff(rating_index);

        // TODO: can be combined in one factor update
        pos_factor_batch.col(num_batched) = cp_v;
        neg_factor_batch.col(num_batched) = cp_v;
        rhs -= (1.0f - score + score * unobserved_weight) * cp_v;

        ++num_batched;
        if (num_batched == batch_size) {
            matrix_symm.rankUpdate(neg_factor_batch, -unobserved_weight);
            matrix_symm.rankUpdate(pos_factor_batch);
            num_batched = 0;
        }
    }
    if (num_batched != 0) {
        auto factor_block = neg_factor_batch.block(0, 0, local_embedding_dim, num_batched);
        matrix_symm.rankUpdate(factor_block, -unobserved_weight);
        factor_block = pos_factor_batch.block(0, 0, local_embedding_dim, num_batched);
        matrix_symm.rankUpdate(factor_block);
    }

    // add "prediction" for the unobserved items
    rhs += unobserved_weight * local_global_gramian * user_embedding;
    // add the regularization.
    rhs += reg * local_user_embedding;

    Eigen::LLT<Eigen::MatrixXf, Eigen::Lower> cholesky(matrix);
    assert(cholesky.info() == Eigen::Success);
    new_value = local_user_embedding - cholesky.solve(rhs);

    return new_value;
}


IALSppRecommender::IALSppRecommender(int embedding_dim, int num_users, int num_items, float reg,
                                     float reg_exp, float unobserved_weight, float stdev,
                                     int block_size)
        : user_embedding_(num_users, embedding_dim),
          item_embedding_(num_items, embedding_dim) {
    // Initialize embedding matrices
    float adjusted_stdev = stdev / sqrt(embedding_dim);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> d(0, adjusted_stdev);
    auto init_matrix = [&](MatrixXf *matrix) {
        for (int i = 0; i < matrix->size(); ++i) {
            *(matrix->data() + i) = d(gen);
        }
    };
    init_matrix(&user_embedding_);
    init_matrix(&item_embedding_);

    regularization_ = reg;
    regularization_exp_ = reg_exp;
    embedding_dim_ = embedding_dim;
    unobserved_weight_ = unobserved_weight;
    block_size_ = std::min(block_size, embedding_dim_);
}

IALSppRecommender::ProjectFunction IALSppRecommender::getProjectFunction() {
    return ProjectBlock;
}


// Custom implementation of EvaluateDataset that does the projection using the
// iterative optimization algorithm.
VectorXf IALSppRecommender::EvaluateDataset(
        const Dataset &data, const SpMatrix &eval_by_user) {
    int num_epochs = 8;

    std::unordered_map<int, VectorXf> user_to_emb;
    VectorXf prediction(data.num_tuples());

    // Initialize the user and predictions to 0.0. (Note: this code needs to
    // change if the embeddings would have biases).
    for (const auto &user_and_history: data.by_user()) {
        user_to_emb[user_and_history.first] = VectorXf::Zero(embedding_dim_);
        for (const auto &item_and_rating_index: user_and_history.second) {
            prediction.coeffRef(item_and_rating_index.second) = 0.0;
        }
    }

    // Train the user embeddings for num_epochs.
    for (int e = 0; e < num_epochs; ++e) {
        // Predict the dataset using the new user embeddings and the existing item
        // embeddings.
        for (const auto &user_and_history: data.by_user()) {
            const VectorXf &user_emb = user_to_emb[user_and_history.first];
            for (const auto &item_and_rating_index: user_and_history.second) {
                prediction.coeffRef(item_and_rating_index.second) =
                        item_embedding_.row(item_and_rating_index.first).dot(user_emb);
            }
        }

        // Optimize the user embeddings for each block.
        for (int start = 0; start < embedding_dim_; start += block_size_) {
            assert(start < embedding_dim_);
            int end = std::min(start + block_size_, embedding_dim_);

            Step(data.by_user(), start, end, &prediction,
                 [&](const int user_id) -> VectorXf & {
                     return user_to_emb[user_id];
                 },
                 item_embedding_,
                    /*index_of_item_bias=*/1);
        }
    }

    // Evalute the dataset.
    return EvaluateDatasetInternal(
            data, eval_by_user,
            [&](const int user_id, const SpVector &history) -> VectorXf {
                return item_embedding_ * user_to_emb[user_id];
            });
}

void IALSppRecommender::Train(const Dataset &data) {
    // Predict the dataset.
    VectorXf prediction(data.num_tuples());
    for (const auto &user_and_history: data.by_user()) {
        VectorXf user_emb = user_embedding_.row(user_and_history.first);
        for (const auto &item_and_rating_index: user_and_history.second) {
            prediction.coeffRef(item_and_rating_index.second) =
                    item_embedding_.row(item_and_rating_index.first).dot(user_emb);
        }
    }

    for (int start = 0; start < embedding_dim_; start += block_size_) {
        assert(start < embedding_dim_);
        int end = std::min(start + block_size_, embedding_dim_);

        Step(data.by_user(), start, end, &prediction,
             [&](const int index) -> MatrixXf::RowXpr {
                 return user_embedding_.row(index);
             },
             item_embedding_,
                /*index_of_item_bias=*/1);
        ComputeLosses(data, prediction);

        // Optimize the item embeddings
        Step(data.by_item(), start, end, &prediction,
             [&](const int index) -> MatrixXf::RowXpr {
                 return item_embedding_.row(index);
             },
             user_embedding_,
                /*index_of_item_bias=*/0);
        ComputeLosses(data, prediction);
    }

//    std::cout << user_embedding_ * item_embedding_.transpose() << "\n\n";

}

void IALSppRecommender::ComputeLosses(const Dataset &data, const VectorXf &prediction) {
    if (!print_trainstats_) {
        return;
    }
    auto time_start = std::chrono::steady_clock::now();
    int num_items = item_embedding_.rows();
    int num_users = user_embedding_.rows();

    // Compute observed loss.
    float loss_observed = (prediction.array() - 1.0).matrix().squaredNorm();

    // Compute regularizer.
    double loss_reg = 0.0;
    for (auto user_and_history: data.by_user()) {
        loss_reg += user_embedding_.row(user_and_history.first).squaredNorm() *
                    RegularizationValue(user_and_history.second.size(), num_items);
    }
    for (auto item_and_history: data.by_item()) {
        loss_reg += item_embedding_.row(item_and_history.first).squaredNorm() *
                    RegularizationValue(item_and_history.second.size(), num_users);
    }

    // Unobserved loss.
    MatrixXf user_gramian = user_embedding_.transpose() * user_embedding_;
    MatrixXf item_gramian = item_embedding_.transpose() * item_embedding_;
    float loss_unobserved = this->unobserved_weight_ * (
            user_gramian.array() * item_gramian.array()).sum();

    // Remove seen interactions
    loss_unobserved -= this->unobserved_weight_ * prediction.squaredNorm();

    float loss = loss_observed + loss_unobserved + loss_reg;

    auto time_end = std::chrono::steady_clock::now();

    printf("Loss=%f, Loss_observed=%f Loss_unobserved=%f Loss_reg=%f Time=%d\n",
           loss, loss_observed, loss_unobserved, loss_reg,
           std::chrono::duration_cast<std::chrono::milliseconds>(
                   time_end - time_start));
}

template<typename F>
void IALSppRecommender::Step(const SpMatrix &data_by_user,
                             const int block_start,
                             const int block_end,
                             VectorXf *prediction,
                             F get_user_embedding_ref,
                             const MatrixXf &item_embedding,
                             const int index_of_item_bias) {
    MatrixXf local_item_emb = item_embedding.block(
            0, block_start, item_embedding.rows(), block_end - block_start);

    // TODO: consider creating the local_gramian as a block from the local_global_gramian
    MatrixXf local_gramian = local_item_emb.transpose() * local_item_emb;
    MatrixXf local_global_gramian = local_item_emb.transpose() * item_embedding;

    // Used for per user regularization.
    int num_items = item_embedding.rows();

    std::mutex m;
    auto Project = getProjectFunction();
    auto data_by_user_iter = data_by_user.begin();  // protected by m
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(std::thread([&] {
            while (true) {
                // Get a new user to work on.
                m.lock();
                if (data_by_user_iter == data_by_user.end()) {
                    m.unlock();
                    return;
                }
                int u = data_by_user_iter->first;
                SpVector train_history = data_by_user_iter->second;
                ++data_by_user_iter;
                m.unlock();

                assert(!train_history.empty());
                float reg = RegularizationValue(train_history.size(), num_items);
                VectorXf old_user_emb = get_user_embedding_ref(u);
                VectorXf old_local_user_emb = old_user_emb.segment(
                        block_start, block_end - block_start);
                VectorXf new_local_user_emb = Project(
                        train_history,
                        old_user_emb,
                        old_local_user_emb,
                        local_item_emb,
                        *prediction,
                        local_gramian,
                        local_global_gramian,
                        reg, this->unobserved_weight_);
                // Update the ratings (without a lock)
                VectorXf delta_local_user_emb =
                        new_local_user_emb - old_local_user_emb;
                for (const auto &item_and_rating_index: train_history) {
                    prediction->coeffRef(item_and_rating_index.second) +=
                            delta_local_user_emb.dot(
                                    local_item_emb.row(item_and_rating_index.first));
                }
                // Update the user embedding.
                m.lock();
                get_user_embedding_ref(u).segment(
                        block_start, block_end - block_start) = new_local_user_emb;
                m.unlock();
            }
        }));
    }
    // Join all threads.
    for (int i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }
}
