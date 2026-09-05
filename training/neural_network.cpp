//
// Created by cartercpp on 9/5/26.
//

#include "neural_network.h"
#include <initializer_list>
#include <vector>
#include <random>
#include <utility>
#include <cstddef>
#include <cmath>
#include "math_vector.h"
#include "matrix.h"

math_vector<double> neural_network::Relu(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = (vec[i] > 0) ? vec[i] : 0;

    return vec;
}

math_vector<double> neural_network::ReluDerivative(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = vec[i] > 0;

    return vec;
}

std::vector<math_vector<double>> neural_network::Forward(const math_vector<double>& input) const
{
    std::vector<math_vector<double>> activations;
    activations.reserve(m_weightMatrices.size() + 1);
    activations.push_back(input);

    for (std::size_t layer = 0; layer < m_weightMatrices.size(); ++layer)
    {
        if (layer + 1 < m_weightMatrices.size())
            activations.emplace_back(Relu(m_weightMatrices[layer] * activations[layer] + m_biasVectors[layer]));
        else
            activations.emplace_back(m_weightMatrices[layer] * activations[layer] + m_biasVectors[layer]);
    }

    return activations;
}

math_vector<double> neural_network::predict(const math_vector<double>& input) const
{
    return Forward(input).back();
}

void neural_network::fit(const math_vector<double>& input, const math_vector<double>& target)
{
    const std::vector<math_vector<double>> activations{Forward(input)};
    const math_vector<double>& prediction{activations.back()};

    math_vector<double> activationDelta = prediction - target;
    std::vector<matrix<double>> weightDeltas(m_weightMatrices.size());
    std::vector<math_vector<double>> biasDeltas(m_biasVectors.size());

    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        const std::size_t layer = m_weightMatrices.size() - 1 - i;
        auto temp = activationDelta;

        if (layer < m_weightMatrices.size() - 1)
            temp = temp.multiply(ReluDerivative(activations[layer + 1]));

        weightDeltas[layer] = outer_product(temp, activations[layer]);
        biasDeltas[layer] = temp;

        activationDelta = m_weightMatrices[layer].transpose() * temp;
    }

    for (std::size_t layer = 0; layer < m_weightMatrices.size(); ++layer)
    {
        m_weightMatrices[layer] -= m_learningRate * weightDeltas[layer];
        m_biasVectors[layer] -= m_learningRate * biasDeltas[layer];
    }
}

neural_network::neural_network(
    std::initializer_list<matrix<double>> weightMatrices,
    std::initializer_list<math_vector<double>> biasVectors,
    std::initializer_list<std::size_t> neuronsPerLayer,
    double learningRate
) : m_weightMatrices{weightMatrices}, m_biasVectors{biasVectors}, m_neuronsPerLayer{neuronsPerLayer},
    m_learningRate{learningRate}
{}

neural_network::neural_network(std::initializer_list<std::size_t> neuronsPerLayer, double learningRate)
    : m_neuronsPerLayer{neuronsPerLayer}, m_learningRate{learningRate}
{
    const std::size_t layers = neuronsPerLayer.size();

    m_weightMatrices.reserve(layers);
    m_biasVectors.reserve(layers);

    std::random_device rd;

    for (std::size_t layer = 1; layer < layers; ++layer)
    {
        const std::size_t prevLayerSize = *(neuronsPerLayer.begin() + layer - 1),
                          layerSize = *(neuronsPerLayer.begin() + layer);

        std::normal_distribution<double> dist(0, std::sqrt(1 / static_cast<double>(2 * prevLayerSize)));

        matrix<double> weights(layerSize, prevLayerSize, 0);
        for (std::size_t i = 0; i < layerSize; ++i)
            for (std::size_t i2 = 0; i2 < prevLayerSize; ++i2)
                weights[i][i2] = dist(rd);

        m_weightMatrices.emplace_back(std::move(weights));
        m_biasVectors.emplace_back(layerSize, 0);
    }
}
