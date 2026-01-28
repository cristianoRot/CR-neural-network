#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <string>
#include <sstream>

class LossGraph {
private:
    std::vector<double> loss_history;
    std::vector<double> accuracy_history;
    double initial_loss = -1.0;
    double min_loss_ever = -1.0;
    double max_accuracy_ever = -1.0;
    size_t graph_width = 80;
    size_t graph_height = 20;
    
    const std::string GRAY = "\033[90m";
    const std::string RED = "\033[31m";
    const std::string BLUE = "\033[34m";
    const std::string RESET = "\033[0m";

    void draw_axes(std::vector<std::vector<char>>& grid, std::vector<std::vector<std::string>>& colors) {
        for (size_t i = 0; i < graph_height; i++) {
            grid[i][0] = '|';
            colors[i][0] = GRAY;
        }
        for (size_t j = 0; j < graph_width; j++) {
            grid[graph_height - 1][j] = '-';
            colors[graph_height - 1][j] = GRAY;
        }
        grid[graph_height - 1][0] = '+';
        colors[graph_height - 1][0] = GRAY;
    }
    
    void plot_loss(std::vector<std::vector<char>>& grid, std::vector<std::vector<std::string>>& colors, 
                   double min_loss, double max_loss, double range) {
        size_t history_size = loss_history.size();
        size_t points_to_plot = std::min(history_size, graph_width);
        
        if (points_to_plot > 1) {
            std::vector<size_t> y_positions(points_to_plot);
            
            for (size_t i = 0; i < points_to_plot; i++) {
                size_t idx = history_size - points_to_plot + i;
                double normalized = (loss_history[idx] - min_loss) / range;
                size_t y = graph_height - 2 - static_cast<size_t>(normalized * (graph_height - 3));
                if (y >= graph_height - 1) y = graph_height - 2;
                if (y < 1) y = 1;
                y_positions[i] = y;
            }
            
            for (size_t i = 0; i < points_to_plot; i++) {
                size_t x = i + 1;
                size_t y = y_positions[i];
                grid[y][x] = '*';
                colors[y][x] = RED;
                
                if (i > 0) {
                    size_t prev_y = y_positions[i - 1];
                    size_t prev_x = i;
                    
                    int dx = x - prev_x;
                    int dy = static_cast<int>(y) - static_cast<int>(prev_y);
                    
                    int steps = std::max(std::abs(dx), std::abs(dy));
                    if (steps > 0) {
                        for (int step = 1; step <= steps; step++) {
                            int interp_x = prev_x + (dx * step) / steps;
                            int interp_y = static_cast<int>(prev_y) + (dy * step) / steps;
                            
                            if (interp_x >= 1 && interp_x < static_cast<int>(graph_width) &&
                                interp_y >= 1 && interp_y < static_cast<int>(graph_height - 1)) {
                                if (grid[interp_y][interp_x] == ' ') {
                                    grid[interp_y][interp_x] = (dx == 0) ? '|' : '-';
                                    colors[interp_y][interp_x] = RED;
                                }
                            }
                        }
                    }
                }
            }
        } else if (points_to_plot == 1) {
            double normalized = (loss_history[0] - min_loss) / range;
            size_t y = graph_height - 2 - static_cast<size_t>(normalized * (graph_height - 3));
            if (y >= graph_height - 1) y = graph_height - 2;
            if (y < 1) y = 1;
            grid[y][1] = '*';
            colors[y][1] = RED;
        }
    }
    
    void plot_accuracy(std::vector<std::vector<char>>& grid, std::vector<std::vector<std::string>>& colors,
                       double min_acc, double max_acc, double range) {
        size_t history_size = accuracy_history.size();
        size_t points_to_plot = std::min(history_size, graph_width);
        
        if (points_to_plot > 1) {
            std::vector<size_t> y_positions(points_to_plot);
            
            for (size_t i = 0; i < points_to_plot; i++) {
                size_t idx = history_size - points_to_plot + i;
                double normalized = (accuracy_history[idx] - min_acc) / range;
                size_t y = graph_height - 2 - static_cast<size_t>(normalized * (graph_height - 3));
                if (y >= graph_height - 1) y = graph_height - 2;
                if (y < 1) y = 1;
                y_positions[i] = y;
            }
            
            for (size_t i = 0; i < points_to_plot; i++) {
                size_t x = i + 1;
                size_t y = y_positions[i];
                
                if (grid[y][x] == ' ' || grid[y][x] == '*') {
                    grid[y][x] = '*';
                    colors[y][x] = BLUE;
                }
                
                if (i > 0) {
                    size_t prev_y = y_positions[i - 1];
                    size_t prev_x = i;
                    
                    int dx = x - prev_x;
                    int dy = static_cast<int>(y) - static_cast<int>(prev_y);
                    
                    int steps = std::max(std::abs(dx), std::abs(dy));
                    if (steps > 0) {
                        for (int step = 1; step <= steps; step++) {
                            int interp_x = prev_x + (dx * step) / steps;
                            int interp_y = static_cast<int>(prev_y) + (dy * step) / steps;
                            
                            if (interp_x >= 1 && interp_x < static_cast<int>(graph_width) &&
                                interp_y >= 1 && interp_y < static_cast<int>(graph_height - 1)) {
                                if (grid[interp_y][interp_x] == ' ') {
                                    grid[interp_y][interp_x] = (dx == 0) ? '|' : '-';
                                    colors[interp_y][interp_x] = BLUE;
                                }
                            }
                        }
                    }
                }
            }
        } else if (points_to_plot == 1) {
            double normalized = (accuracy_history[0] - min_acc) / range;
            size_t y = graph_height - 2 - static_cast<size_t>(normalized * (graph_height - 3));
            if (y >= graph_height - 1) y = graph_height - 2;
            if (y < 1) y = 1;
            if (grid[y][1] == ' ' || grid[y][1] == '*') {
                grid[y][1] = '*';
                colors[y][1] = BLUE;
            }
        }
    }

public:
    LossGraph(size_t width = 60, size_t height = 15) : graph_width(width), graph_height(height) {}

    void set_dimensions(size_t width, size_t height) {
        graph_width = width;
        graph_height = height;
    }

    std::vector<std::string> render_graph(const std::string& title) {
        double current_min_loss = loss_history.empty() ? 0.0 : *std::min_element(loss_history.begin(), loss_history.end());
        double current_max_loss = loss_history.empty() ? 1.0 : *std::max_element(loss_history.begin(), loss_history.end());
        
        if (min_loss_ever < 0 || current_min_loss < min_loss_ever) {
            min_loss_ever = current_min_loss;
        }
        
        double max_scale_loss = initial_loss;
        double min_scale_loss = min_loss_ever;
        
        double padding_loss = (max_scale_loss - min_scale_loss) * 0.05;
        if (padding_loss < 1e-6) padding_loss = 1e-6;
        max_scale_loss += padding_loss;
        min_scale_loss = std::max(0.0, min_scale_loss - padding_loss);
        
        double range_loss = max_scale_loss - min_scale_loss;
        if (range_loss < 1e-10) range_loss = 1.0;
        
        double current_min_acc = accuracy_history.empty() ? 0.0 : *std::min_element(accuracy_history.begin(), accuracy_history.end());
        double current_max_acc = accuracy_history.empty() ? 1.0 : *std::max_element(accuracy_history.begin(), accuracy_history.end());
        
        if (max_accuracy_ever < 0 || current_max_acc > max_accuracy_ever) {
            max_accuracy_ever = current_max_acc;
        }
        
        double max_scale_acc = max_accuracy_ever;
        double min_scale_acc = current_min_acc;
        
        double padding_acc = (max_scale_acc - min_scale_acc) * 0.05;
        if (padding_acc < 1e-6) padding_acc = 1e-6;
        max_scale_acc += padding_acc;
        min_scale_acc = std::max(0.0, min_scale_acc - padding_acc);
        
        double range_acc = max_scale_acc - min_scale_acc;
        if (range_acc < 1e-10) range_acc = 1.0;
        
        std::vector<std::vector<char>> grid(graph_height, std::vector<char>(graph_width, ' '));
        std::vector<std::vector<std::string>> colors(graph_height, std::vector<std::string>(graph_width, RESET));
        
        draw_axes(grid, colors);
        plot_loss(grid, colors, min_scale_loss, max_scale_loss, range_loss);
        plot_accuracy(grid, colors, min_scale_acc, max_scale_acc, range_acc);
        
        std::vector<std::string> output_lines;
        
        // Title line
        std::string title_line = title + ": Loss(" + RED + "Red" + RESET + ") | Acc(" + BLUE + "Blue" + RESET + ")";
        int padding_len = std::max(0, (int)graph_width - (int)title.length() - 25); 
        // Simple centering or just print. Let's just print.
        output_lines.push_back(title_line + std::string(std::max(0, (int)graph_width - (int)title_line.length() + 10), ' ')); 

        // Top Y axis label row
        std::stringstream ss_top;
        ss_top << GRAY << "  Y" << RESET << std::string(graph_width - 2, ' ');
        output_lines.push_back(ss_top.str());
        
        for (size_t i = 0; i < graph_height; i++) {
            std::stringstream line_ss;
            line_ss << "  ";
            for (size_t j = 0; j < graph_width; j++) {
                line_ss << colors[i][j] << grid[i][j] << RESET;
            }
            if (i == 0) {
                 std::stringstream info;
                 info << " " << std::fixed << std::setprecision(4) << max_scale_loss << "/" << max_scale_acc;
                 line_ss << info.str();
            } else if (i == graph_height - 1) {
                 std::stringstream info;
                 info << " " << std::fixed << std::setprecision(4) << min_scale_loss << "/" << min_scale_acc;
                 line_ss << info.str();
            }
            output_lines.push_back(line_ss.str());
        }
        
        std::stringstream ss_bot;
        ss_bot << GRAY << "  " << std::string(graph_width, '-') << " X" << RESET;
        output_lines.push_back(ss_bot.str());

        return output_lines;
    }

    void add_data(double loss, double accuracy) {
        if (initial_loss < 0) {
            initial_loss = loss;
            min_loss_ever = loss;
        }
        
        if (loss < min_loss_ever) {
            min_loss_ever = loss;
        }
        
        if (max_accuracy_ever < 0 || accuracy > max_accuracy_ever) {
            max_accuracy_ever = accuracy;
        }
        
        loss_history.push_back(loss);
        accuracy_history.push_back(accuracy);
        
        if (loss_history.size() > graph_width) {
            loss_history.erase(loss_history.begin(), loss_history.begin() + (loss_history.size() - graph_width));
            accuracy_history.erase(accuracy_history.begin(), accuracy_history.begin() + (accuracy_history.size() - graph_width));
        }
    }
    
    // Legacy support if needed, but we will use render_graph
    void draw() {}
};

class TrainingLogger {
private:
    LossGraph train_graph;
    LossGraph eval_graph;
    double max_train_accuracy = 0.0;
    double max_eval_accuracy = 0.0;
    size_t graph_height = 15;
    const std::string BLUE = "\033[34m";
    const std::string RED = "\033[31m";
    const std::string RESET = "\033[0m";

    void clear_lines(size_t lines) {
        for (size_t i = 0; i < lines; i++) {
            std::cout << "\033[A\033[2K";
        }
    }

public:
    TrainingLogger() : train_graph(50, 15), eval_graph(50, 15) {}

    void log_epoch(size_t epoch, size_t total_epochs, double train_acc, double train_loss, double eval_acc, double eval_loss) {
        if (train_acc > max_train_accuracy) max_train_accuracy = train_acc;
        if (eval_acc > max_eval_accuracy) max_eval_accuracy = eval_acc;
        
        train_graph.add_data(train_loss, train_acc);
        eval_graph.add_data(eval_loss, eval_acc);
        
        // Calculate lines to clear (header + 2 graphs side by side height + bottom)
        // Header: 1 line
        // Graph: render_graph returns title + top Y + height + bottom X = 1 + 1 + height + 1 = height + 3 lines.
        size_t lines_to_render = graph_height + 3;
        clear_lines(1 + lines_to_render); 
        
        std::cout << "\033[1mEpoch " << epoch << "/" << total_epochs 
                  << " | " << BLUE << "Train Acc: " << std::fixed << std::setprecision(4) << train_acc * 100 << "%" << RESET
                  << " | " << RED << "Train Loss: " << train_loss << RESET
                  << " | " << BLUE << "Eval Acc: " << std::fixed << std::setprecision(4) << eval_acc * 100 << "%" << RESET
                  << " | " << RED << "Eval Loss: " << eval_loss << RESET << "\033[0m" << std::endl;
        
        std::vector<std::string> train_lines = train_graph.render_graph("TRAIN");
        std::vector<std::string> eval_lines = eval_graph.render_graph("EVAL");
        
        size_t max_lines = std::max(train_lines.size(), eval_lines.size());
        
        for (size_t i = 0; i < max_lines; i++) {
            std::string left = (i < train_lines.size()) ? train_lines[i] : "";
            std::string right = (i < eval_lines.size()) ? eval_lines[i] : "";
            
            // Pad left to align
            // Note: left contains ANSI codes, so simple length check is improper for visual alignment, 
            // but for simplicity assuming fixed width content mainly.
            // We'll just print a tab or spaces.
            // Since our graph width is fixed (50), we can approximate.
            // But ANSI codes messes up setw.
            // Let's just print simple concatenation with separator.
            
            std::cout << left << "   |   " << right << std::endl;
        }
        std::cout.flush();
    }
    
    void log_completion() {
        std::cout << "\n\033[1;32mTraining completed!\033[0m" << std::endl;
        std::cout << BLUE << "Final Max Train Acc: " << max_train_accuracy * 100 << "%" << RESET << std::endl;
        std::cout << BLUE << "Final Max Eval Acc: " << max_eval_accuracy * 100 << "%" << RESET << std::endl;
    }
};
