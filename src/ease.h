#ifndef LOGMF_EXPERIMENTS_EASE_H
#define LOGMF_EXPERIMENTS_EASE_H

#include "recommender.h"

#include "Eigen/Sparse"

typedef Eigen::SparseMatrix<float , Eigen::RowMajor> MatrixSpf;
typedef Eigen::SparseVector<float , Eigen::RowMajor> VectorSpf;


class EASERecommender : public Recommender {
public:
    EASERecommender(int num_items, float reg, float reg_exp);

    VectorXf Score(const int user_id, const SpVector& user_history) override {
        throw("Function 'Score' is not implemented");
    }

    // Custom implementation of EvaluateDataset that does the projection using the
    // iterative optimization algorithm.
    VectorXf EvaluateDataset(
            const Dataset& data, const SpMatrix& eval_by_user) override ;

    void Train(const Dataset& data) override ;

    const float RegularizationValue(int history_size, int num_choices) const {
        // In EDLAE paper a scalar p / (1-p) is used rather than an exponent.
        return this->regularization_ + pow(
                history_size,
                this->regularization_exp_);
    }

    const MatrixXf& item_embedding() const { return item_embedding_; }

    void SetPrintTrainStats(const bool print_trainstats) {
        print_trainstats_ = print_trainstats;
    }

protected:
    MatrixXf item_embedding_;

    float regularization_;
    float regularization_exp_;
    int embedding_dim_;

    bool print_trainstats_;
};


#endif //LOGMF_EXPERIMENTS_EASE_H
