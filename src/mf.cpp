#include <algorithm>
#include <chrono>
#include <mutex>
#include <random>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Core"

#include "mf.h"
#include "recommender.h"


const VectorXf ProjectMF(
        const SpVector &user_history,
        const VectorXf &user_embedding,
        const MatrixXf &item_embeddings,
        const MatrixXf &gramian,
        const float reg) {
    assert(user_history.size() > 0);

    int embedding_dim = item_embeddings.cols();
    assert(embedding_dim > 0);

    VectorXf new_value(embedding_dim);

    Eigen::MatrixXf matrix = gramian;

    for (int i = 0; i < embedding_dim; ++i) {
        matrix(i, i) += reg;
    }

    Eigen::VectorXf rhs = Eigen::VectorXf::Zero(embedding_dim);
    for (const auto &item_and_rating_index: user_history) {
        const int cp = item_and_rating_index.first;
        assert(cp < item_embeddings.rows());
        const VectorXf cp_v = item_embeddings.row(cp);

        rhs += cp_v;
    }

    Eigen::LLT<Eigen::MatrixXf, Eigen::Lower> cholesky(matrix);
    assert(cholesky.info() == Eigen::Success);
    new_value = cholesky.solve(rhs);

    return new_value;
}


MFRecommender::MFRecommender(int embedding_dim, int num_users, int num_items, float reg, float reg_exp, float stdev)
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
}

// Custom implementation of EvaluateDataset that does the projection using the
// iterative optimization algorithm.
VectorXf MFRecommender::EvaluateDataset(
        const Dataset &data, const SpMatrix &eval_by_user) {
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

    // Reproject the users.
    Step(data.by_user(),
         [&](const int user_id) -> VectorXf & {
             return user_to_emb[user_id];
         },
         item_embedding_);

    // Evalute the dataset.
    return EvaluateDatasetInternal(
            data, eval_by_user,
            [&](const int user_id, const SpVector &history) -> VectorXf {
                return item_embedding_ * user_to_emb[user_id];
            });
}

void MFRecommender::Train(const Dataset &data) {
    Step(data.by_user(),
         [&](const int index) -> MatrixXf::RowXpr {
             return user_embedding_.row(index);
         },
         item_embedding_);
    ComputeLosses(data);

    // Optimize the item embeddings
    Step(data.by_item(),
         [&](const int index) -> MatrixXf::RowXpr {
             return item_embedding_.row(index);
         },
         user_embedding_);
    ComputeLosses(data);

//    std::cout << user_embedding_ * item_embedding_.transpose() << "\n\n";
}

void MFRecommender::ComputeLosses(const Dataset &data) {
    if (!print_trainstats_) {
        return;
    }
    auto time_start = std::chrono::steady_clock::now();
    // Predict the dataset.
    VectorXf prediction(data.num_tuples());
    for (const auto &user_and_history: data.by_user()) {
        VectorXf user_emb = user_embedding_.row(user_and_history.first);
        for (const auto &item_and_rating_index: user_and_history.second) {
            prediction.coeffRef(item_and_rating_index.second) =
                    item_embedding_.row(item_and_rating_index.first).dot(user_emb);
        }
    }
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
    float loss_unobserved = (
            user_gramian.array() * item_gramian.array()).sum();

    loss_unobserved -= prediction.squaredNorm();

    float loss = loss_observed + loss_unobserved + loss_reg;

    auto time_end = std::chrono::steady_clock::now();

    printf("Loss=%f, Loss_observed=%f Loss_unobserved=%f Loss_reg=%f Time=%d\n",
           loss, loss_observed, loss_unobserved, loss_reg,
           std::chrono::duration_cast<std::chrono::milliseconds>(
                   time_end - time_start));
}

template<typename F>
void MFRecommender::Step(const SpMatrix &data_by_user,
                           F get_user_embedding_ref,
                           const MatrixXf &item_embedding) {
    MatrixXf gramian = item_embedding.transpose() * item_embedding;

    // Used for per user regularization.
    int num_items = item_embedding.rows();

    std::mutex m;
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
                VectorXf new_user_emb = ProjectMF(
                        train_history,
                        get_user_embedding_ref(u),
                        item_embedding,
                        gramian,
                        reg);
                // Update the user embedding.
                m.lock();
                get_user_embedding_ref(u) = new_user_emb;
                m.unlock();
            }
        }));
    }
    // Join all threads.
    for (int i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }
}
