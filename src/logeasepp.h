#ifndef LOGMF_EXPERIMENTS_LOGEASEPP_H
#define LOGMF_EXPERIMENTS_LOGEASEPP_H

#include "ialspp.h"

#include "Eigen/Sparse"

typedef Eigen::SparseMatrix<float , Eigen::RowMajor> MatrixSpf;
typedef Eigen::SparseVector<float , Eigen::RowMajor> VectorSpf;


class LogEASEppRecommender : public Recommender {
public:
    LogEASEppRecommender(int num_users, int num_items, float reg,
                         float reg_exp, float unobserved_weight, float stdev, int block_size);

    VectorXf Score(const int user_id, const SpVector& user_history) override {
        throw("Function 'Score' is not implemented");
    }

    // Custom implementation of EvaluateDataset that does the projection using the
    // iterative optimization algorithm.
    VectorXf EvaluateDataset(
            const Dataset& data, const SpMatrix& eval_by_user) override ;

    void Train(const Dataset& data) override ;

    virtual void ComputeLosses(const Dataset& data, const VectorXf& prediction);

    const float RegularizationValue(int history_size, int num_choices) const {
        return this->regularization_ + pow(history_size,this->regularization_exp_);
    }


    template <typename F>
    void Step(const SpMatrix& data_by_user,
              const int block_start,
              const int block_end,
              VectorXf* prediction,
              F get_user_embedding_ref,
              const MatrixSpf &embedding,
              const MatrixXf &gramian);

    const MatrixXf& item_embedding() const { return item_embedding_; }

    void SetPrintTrainStats(const bool print_trainstats) {
        print_trainstats_ = print_trainstats;
    }

protected:
    MatrixSpf user_embedding_;
    MatrixXf item_embedding_;

    float regularization_;
    float regularization_exp_;
    int embedding_dim_;
    float unobserved_weight_;
    int block_size_;

    bool print_trainstats_;
};


#endif //LOGMF_EXPERIMENTS_LOGEASEPP_H
