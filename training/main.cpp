#include <iostream>
#include <fstream>
#include <format>
#include <queue>
#include <tuple>
#include <algorithm>
#include <utility>
#include <random>
#include <thread>
#include <stop_token>
#include "neural_network.h"

template <typename ValueType>
struct std::formatter<std::vector<ValueType>>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const std::vector<ValueType>& vec, std::format_context& ctx) const
    {
        auto out = ctx.out();

        *out++ = '{';

        for (std::size_t i = 0; i < vec.size(); ++i)
        {
            if (i != 0)
                out = std::format_to(out, ", ");

            out = std::format_to(out, "{}", vec[i]);
        }

        *out++ = '}';

        return out;
    }
};

int main()
{
    // maze stuff:
    enum class CELL{START, END, OPEN, CLOSED};
    enum class ACTION{UP, DOWN, LEFT, RIGHT};

    constexpr int rows = 20,
                  columns = 30,
                  actions = 4;

    CELL grid[rows][columns] = {
        { CELL::START, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED },
        { CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN },
        { CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN },
        { CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED },
        { CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN },
        { CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN },
        { CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::CLOSED, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::OPEN, CELL::END }
    };

    auto encode = [&grid](int row, int column)
    {
        math_vector<double> encoding(6, 0);

        encoding[0] = row / static_cast<double>(rows - 1);
        encoding[1] = column / static_cast<double>(columns - 1);
        encoding[2] = (row > 0) && (grid[row - 1][column] != CELL::CLOSED);
        encoding[3] = (row + 1 < rows) && (grid[row + 1][column] != CELL::CLOSED);
        encoding[4] = (column > 0) && (grid[row][column - 1] != CELL::CLOSED);
        encoding[5] = (column + 1 < columns) && (grid[row][column + 1] != CELL::CLOSED);

        return encoding;
    };

    auto takeAction = [&grid](int row, int column, ACTION action)
    {
        int newRow = row,
            newColumn = column;
        double reward = -0.1;

        switch (action)
        {
            case ACTION::UP:
                if ((row > 0) && (grid[row - 1][column] != CELL::CLOSED))
                    newRow = row - 1;
                else
                    reward = -1;

                break;
            case ACTION::DOWN:
                if ((row + 1 < rows) && (grid[row + 1][column] != CELL::CLOSED))
                    newRow = row + 1;
                else
                    reward = -1;

                break;
            case ACTION::LEFT:
                if ((column > 0) && (grid[row][column - 1] != CELL::CLOSED))
                    newColumn = column - 1;
                else
                    reward = -1;

                break;
            case ACTION::RIGHT:
                if ((column + 1 < columns) && (grid[row][column + 1] != CELL::CLOSED))
                    newColumn = column + 1;
                else
                    reward = -1;

                break;
        }

        return std::tuple{newRow, newColumn, reward};
    };

    // essentials:
    neural_network onlineNN({6, 32, 32, 32, 4}, 0.005),
                   targetNN{onlineNN};
    constexpr double gamma = 0.99;

    using State = std::pair<int, int>;
    using Experience = std::tuple<State, ACTION, double, State, bool>; // (state, action, reward, nextState, done)

    std::deque<Experience> replayBuffer;

    // visualization thread:
    {
        std::jthread thr{[&](std::stop_token st) {
            std::random_device rd;
            std::uniform_real_distribution<double> chanceDist(0, 1);
            std::uniform_int_distribution<int> actionDist(0, actions - 1);
            double epsilon = 1;
            int frame = 0;

            while (!st.stop_requested())
            {
                int agentRow = 0,
                    agentColumn = 0,
                    stepsTaken = 0;

                while ((grid[agentRow][agentColumn] != CELL::END) && (stepsTaken < 1000))
                {
                    ACTION action;

                    // choose action to take:
                    if (chanceDist(rd) < epsilon)
                        action = static_cast<ACTION>(actionDist(rd));
                    else
                    {
                        const math_vector<double> input{encode(agentRow, agentColumn)},
                                                  output{onlineNN.predict(input)};

                        int actionIndex = 0;
                        for (int i = 1; i < actions; ++i)
                            if (output[i] > output[actionIndex])
                                actionIndex = i;

                        action = static_cast<ACTION>(actionIndex);
                    }

                    // perform action & store experience in replay buffer:
                    auto [newRow, newColumn, reward] = takeAction(agentRow, agentColumn, action);

                    if (grid[newRow][newColumn] == CELL::END)
                        reward = 100;

                    replayBuffer.emplace_back(
                        std::pair<int, int>{agentRow, agentColumn}, // state
                        action,                                     // action
                        reward,                                     // reward
                        std::pair<int, int>{newRow, newColumn},     // nextState
                        grid[newRow][newColumn] == CELL::END        // done
                    );

                    agentRow = newRow;
                    agentColumn = newColumn;

                    if (replayBuffer.size() > 10'000)
                        replayBuffer.pop_front();

                    // train on replay buffer:
                    const std::size_t samples = std::min<std::size_t>(replayBuffer.size(), 32);
                    std::uniform_int_distribution<std::size_t> bufferIndexDist(0, replayBuffer.size() - 1);

                    for (std::size_t iter = 0; iter < samples; ++iter)
                    {
                        // get random experience:
                        const auto& [state, action, reward, nextState, done] = replayBuffer[bufferIndexDist(rd)];

                        const auto [row, column] = state;
                        const auto [nextRow, nextColumn] = nextState;

                        // make predictions given these states:
                        const math_vector<double> onlinePrediction{onlineNN.predict(encode(row, column))},
                                                  targetPrediction{targetNN.predict(encode(nextRow, nextColumn))};

                        // (fit)
                        double bestFutureValue = std::numeric_limits<double>::lowest();
                        for (int i = 0; i < actions; ++i)
                            if (targetPrediction[i] > bestFutureValue)
                                bestFutureValue = targetPrediction[i];

                        const double desiredQValue = !done ? (reward + gamma * bestFutureValue) : reward;
                        math_vector<double> target{onlinePrediction};
                        target[std::to_underlying(action)] = desiredQValue;

                        onlineNN.fit(encode(row, column), target);
                    }

                    if (frame % 500 == 0)
                        targetNN = onlineNN;

                    ++frame;
                    ++stepsTaken;
                }

                epsilon *= 0.995;
            }
        }};
        std::cin.get();
    }

    std::ofstream weightsFile{"/home/cartercpp/Documents/C++/DeepQNetwork/weights.txt"};
    weightsFile << std::format("{}", targetNN.weights());

    std::ofstream biasesFile{"/home/cartercpp/Documents/C++/DeepQNetwork/biases.txt"};
    biasesFile << std::format("{}", targetNN.biases());
}
