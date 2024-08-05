#include <algorithm>
#include <fstream>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Core"

#include "recommender.h"
#include "logmf.h"


const VectorXf ProjectLogMF(
        const SpVector &user_history,
        const VectorXf &user_embedding,
        const MatrixXf &item_embeddings,
        const MatrixXf &gramian,
        const float reg, const float unobserved_weight) {
    assert(user_history.size() > 0);

    int embedding_dim = item_embeddings.cols();
    assert(embedding_dim > 0);

    Eigen::MatrixXf hessian = 2 * unobserved_weight * gramian;
    for (int i = 0; i < embedding_dim; ++i) {
        hessian(i, i) += reg;
    }

    Eigen::VectorXf rhs = hessian * user_embedding;

    const int kMaxBatchSize = 128;
    const int batch_size = std::min(static_cast<int>(user_history.size()), kMaxBatchSize);
    auto matrix_symm = hessian.selfadjointView<Eigen::Lower>();

    int num_batched = 0;
    Eigen::MatrixXf pos_factor_batch(embedding_dim, batch_size);
    Eigen::MatrixXf neg_factor_batch(embedding_dim, batch_size);
    for (const auto &item_and_rating_index: user_history) {
        const int cp = item_and_rating_index.first;
        assert(cp < item_embeddings.rows());
        const VectorXf cp_v = item_embeddings.row(cp);

        float score = user_embedding.dot(cp_v);
        float pred = sigmoid(score);

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
        auto factor_block = neg_factor_batch.block(0, 0, embedding_dim, num_batched);
        matrix_symm.rankUpdate(factor_block, -2 * unobserved_weight);
        factor_block = pos_factor_batch.block(0, 0, embedding_dim, num_batched);
        matrix_symm.rankUpdate(factor_block);
    }

    Eigen::LLT<Eigen::MatrixXf, Eigen::Lower> cholesky(hessian);
    assert(cholesky.info() == Eigen::Success);
    VectorXf update_value = cholesky.solve(rhs);

    return user_embedding - update_value;
}

IALSRecommender::ProjectFunction LogMFRecommender::getProjectFunction() {
    return ProjectLogMF;
}

void LogMFRecommender::ComputeLosses(const Dataset &data) {
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
