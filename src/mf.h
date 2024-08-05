#ifndef LOGMF_EXPERIMENTS_MF_H
#define LOGMF_EXPERIMENTS_MF_H

#include "recommender.h"


class MFRecommender : public Recommender {
public:
    MFRecommender(int embedding_dim, int num_users, int num_items, float reg, float reg_exp, float stdev);

    VectorXf Score(const int user_id, const SpVector &user_history) override {
        throw ("Function 'Score' is not implemented");
    }

    // Custom implementation of EvaluateDataset that does the projection using the
    // iterative optimization algorithm.
    VectorXf EvaluateDataset(const Dataset &data, const SpMatrix &eval_by_user) override;

    void Train(const Dataset &data) override;

    virtual void ComputeLosses(const Dataset &data);

    const float RegularizationValue(int history_size, int num_choices) const {
        return this->regularization_ * pow(history_size,this->regularization_exp_);
    }

    template<typename F>
    void Step(const SpMatrix &data_by_user, F get_user_embedding_ref, const MatrixXf &item_embedding);

    const MatrixXf &item_embedding() const {
        return item_embedding_;
    }

    void SetPrintTrainStats(const bool print_trainstats) {
        print_trainstats_ = print_trainstats;
    }

protected:
    MatrixXf user_embedding_;
    MatrixXf item_embedding_;

    float regularization_;
    float regularization_exp_;
    int embedding_dim_;

    bool print_trainstats_;
};


#endif //LOGMF_EXPERIMENTS_MF_H
