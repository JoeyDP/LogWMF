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

#include "ease.h"
#include "recommender.h"


EASERecommender::EASERecommender(int num_items, float reg, float reg_exp)
        : item_embedding_(num_items, num_items) {
    // Initialize embedding matrices

    regularization_ = reg;
    regularization_exp_ = reg_exp;
    embedding_dim_ = num_items;
}


// Custom implementation of EvaluateDataset that does the projection using the
// iterative optimization algorithm.
VectorXf EASERecommender::EvaluateDataset(const Dataset &data, const SpMatrix &eval_by_user) {
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

void EASERecommender::Train(const Dataset &data) {
    Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");

    // Set user embedding
    int num_users = data.max_user()+1;
    MatrixSpf X(num_users, embedding_dim_);
    printf("Set User emb\n");
    typedef Eigen::Triplet<uint8_t> T;
    std::vector<T> tripletList;
    tripletList.reserve(data.num_tuples());
    for (const auto &row: data.by_user()) {
        for (const auto &entry: row.second) {
            tripletList.emplace_back(row.first, entry.first, 1);
        }
    }

    X.setFromTriplets(tripletList.begin(), tripletList.end());
    X.makeCompressed();

    MatrixXf XTX = (X.transpose().cast<uint32_t>() * X.cast<uint32_t>()).cast<float>();

    for (int i = 0; i < embedding_dim_; i++) {
        XTX.coeffRef(i, i) += RegularizationValue((int) X.col(i).sum(), num_users);
    }

//    std::cout << XTX << "\n\n";

    Eigen::Inverse<MatrixXf> P(XTX);

    // Item embedding is transpose of square matrix B from EASE
    item_embedding_ = -(P * P.diagonal().cwiseInverse().asDiagonal()).transpose();
    item_embedding_.diagonal().setZero();

    // Print model
//    std::cout << item_embedding_.format(HeavyFmt) << "\n\n";
}

