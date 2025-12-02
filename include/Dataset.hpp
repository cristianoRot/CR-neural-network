// dataset.hpp

#pragma once
#include "Matrix.hpp"
#include <vector>
#include <string>

class Dataset 
{
    private:   
        std::vector<Matrix> inputs;
        std::vector<size_t> outputs;

        std::vector<size_t> perm_idx;

    public:
        Dataset(std::vector<Matrix> inputs, std::vector<size_t> outputs);

        const size_t size() const;
        
        Matrix get_input(size_t start_idx, size_t batch_size) const;
        std::vector<size_t> get_output(size_t start_idx, size_t batch_size) const;

        void shuffle();
        
        static Dataset from_csv(
            const std::string& file_path,
            const std::vector<std::string>& input_columns,
            const std::string& output_column,
            size_t start_row = 0,
            size_t end_row = SIZE_MAX
        );
};
