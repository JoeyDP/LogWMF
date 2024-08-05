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

#include "easepp.h"
#include "recommender.h"


const VectorXf ProjectEASEBlock(
        const SpVector &user_history,
        const VectorXf &user_embedding,
        const VectorXf &local_user_embedding,
        const MatrixSpf &local_item_embedding,
        const VectorXf &prediction,
        const MatrixXf &local_gramian,
        const MatrixXf &local_global_gramian,
        const float reg, const float unobserved_weight,
        int diagonalIndex) {
    assert(user_history.size() > 0);
    int local_embedding_dim = local_item_embedding.cols();
    assert(local_embedding_dim > 0);

    VectorXf new_value(local_embedding_dim);

    MatrixXf matrix = unobserved_weight * local_gramian;

    for (int i = 0; i < local_embedding_dim; ++i) {
        matrix(i, i) += reg;
    }

    const int kMaxBatchSize = 128;
    auto matrix_symm = matrix.selfadjointView<Eigen::Lower>();
    VectorXf rhs = Eigen::VectorXf::Zero(local_embedding_dim);
    const int batch_size = std::min(static_cast<int>(user_history.size()), kMaxBatchSize);
    int num_batched = 0;
    MatrixXf pos_factor_batch(local_embedding_dim, batch_size);
    MatrixXf neg_factor_batch(local_embedding_dim, batch_size);
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

    // Constrained Newton Method
    if (diagonalIndex >= 0 && diagonalIndex < local_embedding_dim){
        // Expand matrix
        matrix.conservativeResize(matrix.rows() + 1, matrix.cols() + 1);

        // Set index element (only lower triangular needs to be updated)
        matrix.row(matrix.rows()-1) = VectorXf::Zero(matrix.rows());
        matrix.coeffRef(matrix.rows()-1, diagonalIndex) = 1;

        rhs.conservativeResize(rhs.size() + 1);
        rhs.coeffRef(rhs.size()-1) = 0;
    }

    Eigen::LDLT<MatrixXf, Eigen::Lower> cholesky(matrix);
    assert(cholesky.info() == Eigen::Success);

    VectorXf delta = cholesky.solve(rhs);

    if (diagonalIndex >= 0 && diagonalIndex < local_embedding_dim) {
        if (abs(delta.coeff(diagonalIndex)) > 0.0001){
            printf("Warning: diagonal value would be: %f\n", delta.coeff(diagonalIndex));
        }
//        assert(abs(delta.coeff(diagonalIndex)) < 0.0001);
        // account for rounding errors
        delta.coeffRef(diagonalIndex) = 0;
        delta.conservativeResize(delta.rows() - 1);
    }

    return local_user_embedding - delta;
}


EASEppRecommender::EASEppRecommender(int num_users, int num_items, float reg,
                                     float reg_exp, float unobserved_weight,
                                     float stdev, int block_size)
        : user_embedding_(num_users, num_items),
          item_embedding_(num_items, num_items) {
    // Initialize embedding matrices
    float adjusted_stdev = stdev / sqrt(num_items);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> d(0, adjusted_stdev);
    auto init_matrix = [&](MatrixXf *matrix) {
        for (int i = 0; i < matrix->size(); ++i) {
            *(matrix->data() + i) = d(gen);
        }
    };
    init_matrix(&item_embedding_);
    item_embedding_.diagonal().setZero();

    regularization_ = reg;
    regularization_exp_ = reg_exp;
    embedding_dim_ = num_items;
    unobserved_weight_ = unobserved_weight;
    block_size_ = std::min(block_size, embedding_dim_);
}


// Custom implementation of EvaluateDataset that does the projection using the
// iterative optimization algorithm.
VectorXf EASEppRecommender::EvaluateDataset(const Dataset &data, const SpMatrix &eval_by_user) {
    // Evalute the dataset.
    return EvaluateDatasetInternal(
            data, eval_by_user,
            [&](const int user_id, const SpVector &history) -> VectorXf {
                VectorSpf user(embedding_dim_);
                user.reserve(history.size());
                for (const auto &item: history)
                    user.coeffRef(item.first) = 1;
                return item_embedding_ * user.transpose();
            });

}

void EASEppRecommender::ComputeLosses(const Dataset &data, const VectorXf &prediction) {
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
    MatrixXf user_gramian = (user_embedding_.transpose().cast<uint32_t>() *
                             user_embedding_.cast<uint32_t>()).cast<float>();
    MatrixXf item_gramian = item_embedding_.transpose() * item_embedding_;
    float loss_unobserved = this->unobserved_weight_ * (user_gramian.array() * item_gramian.array()).sum();

    // Remove seen interactions
    loss_unobserved -= this->unobserved_weight_ * prediction.squaredNorm();

    float loss = loss_observed + loss_unobserved + loss_reg;

    auto time_end = std::chrono::steady_clock::now();

    printf("Loss=%f, Loss_observed=%f Loss_unobserved=%f Loss_reg=%f Time=%d\n",
           loss, loss_observed, loss_unobserved, loss_reg,
           std::chrono::duration_cast<std::chrono::milliseconds>(
                   time_end - time_start));
}

void EASEppRecommender::Train(const Dataset &data) {
    // Set user embedding
    if (user_embedding_.nonZeros() == 0) {
        printf("Set User emb\n");
        typedef Eigen::Triplet<uint8_t> T;
        std::vector<T> tripletList;
        tripletList.reserve(data.num_tuples());
        for (const auto &row: data.by_user()) {
            for (const auto &entry: row.second) {
                tripletList.emplace_back(row.first, entry.first, 1);
            }
        }

        user_embedding_.setFromTriplets(tripletList.begin(), tripletList.end());
        user_embedding_.makeCompressed();
    }

    // Predict the dataset.
    VectorXf prediction(data.num_tuples());
    for (const auto &user_and_history: data.by_user()) {
        MatrixSpf::RowXpr user_emb = user_embedding_.row(user_and_history.first);
        for (const auto &item_and_rating_index: user_and_history.second) {
            prediction.coeffRef(item_and_rating_index.second) = user_emb.dot(
                    item_embedding_.row(item_and_rating_index.first));
        }
    }

    MatrixXf gramian = (user_embedding_.transpose().cast<uint32_t>() * user_embedding_.cast<uint32_t>()).cast<float>();

    for (int start = 0; start < embedding_dim_; start += block_size_) {
        assert(start < embedding_dim_);
        int end = std::min(start + block_size_, embedding_dim_);

//        printf("block: %d/%d\n", start, embedding_dim_);

        // Optimize the item embeddings
        Step(data.by_item(), start, end, &prediction,
             [&](const int index) -> MatrixXf::RowXpr {
                 return item_embedding_.row(index);
             },
             user_embedding_,
             gramian);

        // Debug print weights of diag
//        for (int i = start; i < end; i++)
//            printf("diag weights %f\n", item_embedding_.coeff(i, i));

        ComputeLosses(data, prediction);
    }

    // Print model
//    Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
//    std::cout << item_embedding_.format(HeavyFmt) << "\n\n";
}

template<typename F>
void EASEppRecommender::Step(const SpMatrix &data_by_user,
                             const int block_start,
                             const int block_end,
                             VectorXf *prediction,
                             F get_user_embedding_ref,
                             const MatrixSpf &embedding,
                             const MatrixXf &gramian) {

    MatrixSpf local_item_emb = embedding.block(
            0, block_start, embedding.rows(), block_end - block_start);

    int block_size = block_end - block_start;
    MatrixXf local_gramian = gramian.block(block_start, block_start, block_size, block_size);
    MatrixXf local_global_gramian = gramian.block(block_start, 0, block_size, gramian.cols());

    int INNER_ITERATIONS = 1;

    // Used for per user regularization.
    int num_items = gramian.rows();

    std::mutex m;
    auto Project = ProjectEASEBlock;
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


                for(int j = 0; j < INNER_ITERATIONS;j++) {
                    int diagonalIndex = u - block_start;

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
                            reg, this->unobserved_weight_, diagonalIndex);

                    // Update the ratings (without a lock)
                    VectorXf delta_local_user_emb =
                            new_local_user_emb - old_local_user_emb;
                    for (const auto &item_and_rating_index: train_history) {
                        prediction->coeffRef(item_and_rating_index.second) +=
                                local_item_emb.row(item_and_rating_index.first).dot(delta_local_user_emb);
                    }
                    // Update the user embedding.
                    m.lock();
                    get_user_embedding_ref(u).segment(
                            block_start, block_end - block_start) = new_local_user_emb;
                    m.unlock();
                }
            }
        }));
    }
    // Join all threads.
    for (int i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }
}
