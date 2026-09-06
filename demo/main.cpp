#include <iostream>
#include <fstream>
#include <format>
#include <chrono>
#include <thread>
#include <stop_token>
#include "neural_network.h"

extern neural_network mazeNN;

int main()
{
    // maze stuff:
    enum class CELL{START, END, OPEN, CLOSED};
    enum class ACTION{UP, DOWN, LEFT, RIGHT};

    constexpr int rows = 10,
              columns = 15,
              actions = 4;

    CELL grid[rows][columns] = {
        { CELL::START, CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN },
        { CELL::OPEN,  CELL::CLOSED, CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::CLOSED, CELL::CLOSED, CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN,   CELL::OPEN },
        { CELL::OPEN,  CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::OPEN },
        { CELL::CLOSED,CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN },
        { CELL::OPEN,  CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::CLOSED, CELL::CLOSED, CELL::OPEN },
        { CELL::OPEN,  CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN },
        { CELL::OPEN,  CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::OPEN },
        { CELL::OPEN,  CELL::CLOSED, CELL::CLOSED, CELL::CLOSED, CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN },
        { CELL::OPEN,  CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::CLOSED, CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN },
        { CELL::CLOSED,CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::CLOSED, CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::OPEN,   CELL::END }
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

    // visualization:
    std::cout << "\033[?25l"; // cursor off
    std::cout << "\033[2J"; // clear screen

    {
        std::jthread thr{[&](std::stop_token st) {
            while (!st.stop_requested())
            {
                int agentRow = 0,
                    agentColumn = 0;

                while (true)
                {
                    std::cout << "\033[0H";
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int column = 0; column < columns; ++column)
                            switch (grid[row][column])
                            {
                                case CELL::START:
                                    std::cout << "\033[38;2;255;255;0m_";
                                    break;
                                case CELL::OPEN:
                                    std::cout << "\033[38;2;255;255;255m ";
                                    break;
                                case CELL::CLOSED:
                                    std::cout << "\033[38;2;255;0;0m█";
                                    break;
                                case CELL::END:
                                    std::cout << "\033[38;2;0;255;0m_";
                                    break;
                            }

                        std::cout << '\n';
                    }
                    std::cout << std::format("\033[{};{}H", agentRow + 1, agentColumn + 1);
                    std::cout << "\033[38;2;0;255;255mo" << std::flush;

                    if (grid[agentRow][agentColumn] == CELL::END)
                    {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        break;
                    }

                    const auto qValues = mazeNN.predict(encode(agentRow, agentColumn));
                    int actionIndex = 0;

                    for (int i = 1; i < actions; ++i)
                        if (qValues[i] > qValues[actionIndex])
                            actionIndex = i;

                    const ACTION action = static_cast<ACTION>(actionIndex);

                    switch (action)
                    {
                        case ACTION::UP:
                            if ((agentRow > 0) && (grid[agentRow - 1][agentColumn] != CELL::CLOSED))
                                --agentRow;

                            break;
                        case ACTION::DOWN:
                            if ((agentRow + 1 < rows) && (grid[agentRow + 1][agentColumn] != CELL::CLOSED))
                                ++agentRow;

                            break;
                        case ACTION::LEFT:
                            if ((agentColumn > 0) && (grid[agentRow][agentColumn - 1] != CELL::CLOSED))
                                --agentColumn;

                            break;
                        case ACTION::RIGHT:
                            if ((agentColumn + 1 < columns) && (grid[agentRow][agentColumn + 1] != CELL::CLOSED))
                                ++agentColumn;

                            break;
                    }

                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        }};
        std::cin.get();
    }

    std::cout << "\033[?25h"; // restore cursor
}
