#ifndef LOGMF_EXPERIMENTS_LOGMFPP_H
#define LOGMF_EXPERIMENTS_LOGMFPP_H

#include "ialspp.h"


class LogMFppRecommender : public IALSppRecommender {
public:
    LogMFppRecommender(int embedding_dim, int num_users, int num_items, float reg,
                     float reg_exp, float unobserved_weight, float stdev, int block_size)
                     : IALSppRecommender(embedding_dim, num_users, num_items, reg, reg_exp, unobserved_weight, stdev, block_size){}

    IALSppRecommender::ProjectFunction getProjectFunction() override;

    void ComputeLosses(const Dataset &data, const VectorXf &prediction) override;
};


#endif //LOGMF_EXPERIMENTS_LOGMFPP_H
