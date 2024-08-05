#ifndef LOGMF_EXPERIMENTS_LOGMF_H
#define LOGMF_EXPERIMENTS_LOGMF_H

#include "ials.h"

class LogMFRecommender : public IALSRecommender {
public:
    LogMFRecommender(int embedding_dim, int num_users, int num_items, float reg, float reg_exp, float unobserved_weight, float stdev)
                     : IALSRecommender(embedding_dim, num_users, num_items, reg, reg_exp, unobserved_weight, stdev){}

    IALSRecommender::ProjectFunction getProjectFunction() override;

    void ComputeLosses(const Dataset &data) override;

};


inline MatrixXf sigmoid(const MatrixXf &m){
    return 1 / (1 + (-m.array()).exp());
}

inline float sigmoid(float f){
    return 1 / (1 + std::exp(-f));
}


#endif //LOGMF_EXPERIMENTS_LOGMF_H
