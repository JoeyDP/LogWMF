#include <chrono>
#include <fstream>
#include <mutex>
#include <set>
#include <stdlib.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Core"

#include "ease.h"


int main(int argc, char *argv[]) {
    // Default flags.
    std::unordered_map<std::string, std::string> flags;
    flags["embedding_dim"] = "0";
    flags["unobserved_weight"] = "1";
    flags["regularization"] = "0.0001";
    flags["regularization_exp"] = "1.0";
    flags["stddev"] = "0";
    flags["print_train_stats"] = "0";
    flags["block_size"] = "0";
    flags["eval_during_training"] = "1";
    flags["results_file"] = "results.csv";
    flags["epochs"] = "1";

    // Parse flags. This is a simple implementation to avoid external
    // dependencies.
    for (int i = 1; i < argc; ++i) {
        assert(i < (argc - 1));
        std::string flag_name = argv[i];
        assert(flag_name.at(0) == '-');
        if (flag_name.at(1) == '-') {
            flag_name = flag_name.substr(2);
        } else {
            flag_name = flag_name.substr(1);
        }
        ++i;
        std::string flag_value = argv[i];
        flags[flag_name] = flag_value;
    }

    // Data related flags must exist.
    assert(flags.count("train_data") == 1);
    assert(flags.count("test_train_data") == 1);
    assert(flags.count("test_test_data") == 1);

    // Load the datasets
    Dataset train(flags.at("train_data"));
    Dataset test_tr(flags.at("test_train_data"));
    Dataset test_te(flags.at("test_test_data"));

    // Create the recommender.
    Recommender *recommender;
    recommender = new EASERecommender(
            train.max_item() + 1,
            std::atof(flags.at("regularization").c_str()),
            std::atof(flags.at("regularization_exp").c_str())
            );
    ((EASERecommender *) recommender)->SetPrintTrainStats(
            std::atoi(flags.at("print_train_stats").c_str()));

    // Disable output buffer to see results without delay.
    setbuf(stdout, NULL);

    bool eval_during_training =
            std::atoi(flags.at("eval_during_training").c_str());

    // Train and evaluate.
    auto time_start = std::chrono::steady_clock::now();
    int num_epochs = std::atoi(flags.at("epochs").c_str());

    // Helper for evaluation.
    auto evaluate = [&](int epoch) {
        auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - time_start);
        VectorXf metrics =
                recommender->EvaluateDataset(test_tr, test_te.by_user());
        printf("Epoch %4d:\t Rec20=%.4f, Rec50=%.4f NDCG100=%.4f\n",
               epoch, metrics[0], metrics[1], metrics[2]);
        if (epoch == num_epochs) {
            Recommender::writeOutput(flags["results_file"], "ease", flags, metrics, train_time);
        }
    };


    auto time_train_start = std::chrono::steady_clock::now();
    recommender->Train(train);
    auto time_train_end = std::chrono::steady_clock::now();
    auto time_eval_start = std::chrono::steady_clock::now();

    auto time_eval_end = std::chrono::steady_clock::now();
    printf("Timer: Train=%d\tEval=%d\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(
                   time_train_end - time_train_start),
           std::chrono::duration_cast<std::chrono::milliseconds>(
                   time_eval_end - time_eval_start));

    evaluate(num_epochs);


    delete recommender;
    return 0;
}
