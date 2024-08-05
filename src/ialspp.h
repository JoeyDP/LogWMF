#ifndef LOGMF_EXPERIMENTS_IALSPP_H
#define LOGMF_EXPERIMENTS_IALSPP_H


#include "recommender.h"


class IALSppRecommender : public Recommender {
public:
    IALSppRecommender(int embedding_dim, int num_users, int num_items, float reg,
                      float reg_exp, float unobserved_weight, float stdev,
                      int block_size);

    VectorXf Score(const int user_id, const SpVector& user_history) override {
        throw("Function 'Score' is not implemented");
    }

    typedef std::function<const VectorXf(const SpVector &, const VectorXf &, const VectorXf &,
                                         const MatrixXf &,const VectorXf &,const MatrixXf &,
                                         const MatrixXf &, const float , const float)> ProjectFunction;
    virtual ProjectFunction getProjectFunction();

    // Custom implementation of EvaluateDataset that does the projection using the
    // iterative optimization algorithm.
    VectorXf EvaluateDataset(
            const Dataset& data, const SpMatrix& eval_by_user) override ;

    void Train(const Dataset& data) override ;

    virtual void ComputeLosses(const Dataset& data, const VectorXf& prediction);

    // Computes the regularization value for a user (or item). The value depends
    // on the number of observations for this user (or item) and the total number
    // of items (or users).
    const float RegularizationValue(int history_size, int num_choices) const {
        return this->regularization_ * pow(history_size,this->regularization_exp_);
//        return this->regularization_ * pow(
//                history_size + this->unobserved_weight_ * num_choices,
//                this->regularization_exp_);
    }

    template <typename F>
    void Step(const SpMatrix& data_by_user,
              const int block_start,
              const int block_end,
              VectorXf* prediction,
              F get_user_embedding_ref,
              const MatrixXf& item_embedding,
              const int index_of_item_bias);

    const MatrixXf& item_embedding() const { return item_embedding_; }

    void SetPrintTrainStats(const bool print_trainstats) {
        print_trainstats_ = print_trainstats;
    }

protected:
    MatrixXf user_embedding_;
    MatrixXf item_embedding_;

    float regularization_;
    float regularization_exp_;
    int embedding_dim_;
    float unobserved_weight_;
    int block_size_;

    bool print_trainstats_;
};


#endif //LOGMF_EXPERIMENTS_IALSPP_H
