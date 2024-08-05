#include <algorithm>
#include <fstream>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Core"

#include "logmfpp.h"
#include "logmf.h"


const VectorXf ProjectLogMFBlock(
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

    Eigen::MatrixXf matrix = 2 * unobserved_weight * local_gramian;

    for (int i = 0; i < local_embedding_dim; ++i) {
        matrix(i, i) += reg;
    }

    const int kMaxBatchSize = 128;
    auto matrix_symm = matrix.selfadjointView<Eigen::Lower>();
    Eigen::VectorXf rhs = Eigen::VectorXf::Zero(local_embedding_dim);
    const int batch_size = std::min(static_cast<int>(user_history.size()), kMaxBatchSize);
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
        float pred = sigmoid(score);

//        factor_batch.col(num_batched) = cp_v;
        pos_factor_batch.col(num_batched) = std::sqrt(pred * (1 - pred)) * cp_v;
        neg_factor_batch.col(num_batched) = cp_v;
        rhs -= (1 - pred + 2 * score * unobserved_weight) * cp_v;

        ++num_batched;
        if (num_batched == batch_size) {
            matrix_symm.rankUpdate(neg_factor_batch, -2 * unobserved_weight);
            matrix_symm.rankUpdate(pos_factor_batch);
            num_batched = 0;
        }
    }
    if (num_batched != 0) {
        auto factor_block = neg_factor_batch.block(0, 0, local_embedding_dim, num_batched);
        matrix_symm.rankUpdate(factor_block, -2 * unobserved_weight);
        factor_block = pos_factor_batch.block(0, 0, local_embedding_dim, num_batched);
        matrix_symm.rankUpdate(factor_block);
    }

    // add "prediction" for the unobserved items
    rhs += 2 * unobserved_weight * local_global_gramian * user_embedding;
    // add the regularization.
    rhs += reg * local_user_embedding;

    Eigen::LLT<Eigen::MatrixXf, Eigen::Lower> cholesky(matrix);
    assert(cholesky.info() == Eigen::Success);
    new_value = local_user_embedding - cholesky.solve(rhs);

    return new_value;
}

IALSppRecommender::ProjectFunction LogMFppRecommender::getProjectFunction() {
    return ProjectLogMFBlock;
}

void LogMFppRecommender::ComputeLosses(const Dataset &data, const VectorXf &prediction) {
    if (!print_trainstats_) {
        return;
    }
    auto time_start = std::chrono::steady_clock::now();
    int num_items = item_embedding_.rows();
    int num_users = user_embedding_.rows();

    // Compute observed loss.
    float loss_observed = -(sigmoid(prediction).array()).log().sum();

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