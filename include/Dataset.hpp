// dataset.hpp

#pragma once
#include "Matrix.hpp"
#include <vector>
#include <string>
#include <map>

// ============================================================================
// Dataset Class
// ============================================================================

class Dataset 
{
    private:   
        std::vector<Matrix> inputs;
        std::vector<size_t> outputs;
        std::vector<size_t> perm_idx;

    public:
        // ====================================================================
        // Public Methods
        // ====================================================================
        
        Dataset(std::vector<Matrix> inputs, std::vector<size_t> outputs);

        const size_t size() const;

        Matrix get_input(size_t start_idx, size_t batch_size) const;
        std::vector<size_t> get_output(size_t start_idx, size_t batch_size) const;

        void shuffle();
        
        std::vector<double> get_class_weight() const;

        void scale(double factor);
        
        // ====================================================================
        // Static Methods
        // ====================================================================
        
        static Dataset from_csv(
            const std::string& file_path,
            const std::vector<std::string>& input_columns,
            const std::string& output_column,
            size_t start_row = 0,
            size_t end_row = SIZE_MAX
        );
};

// ============================================================================
// Helper Functions
// ============================================================================

std::string trim_whitespace(const std::string& s);
std::vector<std::string> split(const std::string& s, char delimiter);
size_t validate_batch_size(size_t start_idx, size_t batch_size, size_t dataset_size);
size_t parse_output_value(const std::string& value, std::map<std::string, size_t>& class_map, size_t& next_class_index);
std::vector<size_t> find_input_column_indices(
    const std::vector<std::string>& headers,
    const std::vector<std::string>& input_columns,
    const std::string& output_column
);
