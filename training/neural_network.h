//
// Created by cartercpp on 9/5/26.
//

#ifndef DEEPQNETWORK_NEURAL_NETWORK_H
#define DEEPQNETWORK_NEURAL_NETWORK_H

#include <vector>
#include "math_vector.h"
#include "matrix.h"
#include <cstddef>

class neural_network
{
public:

    // CONSTRUCTORS

    explicit neural_network(
        const std::vector<std::size_t>&,
        double
    );

    explicit neural_network(
        const std::vector<matrix<double>>&,
        const std::vector<math_vector<double>>&,
        const std::vector<std::size_t>&,
        double
    );

    // METHODS

    math_vector<double> predict(const math_vector<double>&) const;
    void fit(const math_vector<double>&, const math_vector<double>&);

    const auto& weights() const
    {
        return m_weightMatrices;
    }

    const auto& biases() const
    {
        return m_biasVectors;
    }

private:

    static math_vector<double> Relu(math_vector<double>);
    static math_vector<double> ReluDerivative(math_vector<double>);

    std::vector<math_vector<double>> Forward(const math_vector<double>&) const;

    std::vector<matrix<double>> m_weightMatrices;
    std::vector<math_vector<double>> m_biasVectors;
    std::vector<std::size_t> m_neuronsPerLayer;
    double m_learningRate;
};

#endif //DEEPQNETWORK_NEURAL_NETWORK_H