#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

bool check_sum(std::vector<int> &array, const int targetSum, int &i, int &j) {
    int N = array.size();
    std::sort(array.begin(), array.end());
    
    int sum = array[N - 1] + array[N - 2];
    if (sum < targetSum) {
        return false;
    } else {
        i = 0;
        j = 1;
        for (int k = 0; k < N; ++k) {
            sum = array[N - j] + array[i];
            if (sum == targetSum) {
                return true;
            } else if (sum < targetSum) {
                ++i;
            } else if (sum > targetSum) {
                ++j;
            }
        }
    }
    
    return false;
}

int main(int argc, char *argv[]) {
    int targetSum;
    int N;
    if (argc > 1) {
        targetSum = atof(argv[1]);
        N = atof(argv[2]);
    } else {
        std::cout << "Enter the target sum: ";
        std::cin >> targetSum;
        std::cout << "Enter the size of the list: ";
        std::cin >> N;
    }
    
    std::random_device seeder;
    std::mt19937_64 gen(seeder());
    std::uniform_int_distribution<int> dist(1, 200);
    
    std::vector<int> array;
    array.reserve(N);
//     std::cout << "array = (";
    for (int i = 0; i < N; ++i) {
        array.push_back(dist(gen));
//         if (i < N - 1) std::cout << array[i] << ", ";
//         else std::cout << array[i] << ")" << std::endl;
    }
    
    int i, j;
    if (check_sum(array, targetSum, i, j)) {
        std::cout << "Found target sum:" << std::endl;
        std::cout << " array[" << i << "] = " << array[i] << std::endl;
        std::cout << " array[" << N - j << "] = " << array[N - j] << std::endl;
        std::cout << " " << array[i] << " + " << array[N - j] << " = " << array[i] + array[N - j] << std::endl;
        std::cout << " targetSum = " << targetSum << std::endl;
    } else {
        std::cout << "No two numbers in the array add up to the target sum." << std::endl;
    }
    
    return 0;
}
    
