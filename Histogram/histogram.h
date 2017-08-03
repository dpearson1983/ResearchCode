#ifndef _HISTOGRAM_H_
#define _HISTOGRAM_H_

double get_interquartile_range(std::vector<double> &data) {
    std::sort(data.begin(), data.end());
    int N = data.size();
    double IQR;
    if (N % 2 == 0) {
        double Q1 = (data[N/4] + data[N/4 - 1])/2.0;
        double Q3 = (data[3*N/4] + data[3*N/4 - 1])/2.0;
        IQR = Q3 - Q1;
    } else {
        double Q1 = data[N/4];
        double Q3 = data[3*N/4];
        IQR = Q3 - Q1;
    }
    return IQR;
}

void histogram(std::vector<double> &bins, std::vector<int> &hist, std::vector<double> &data) {
    double IQR = get_interquartile_range(data);
    double min = data[0];
    double max = data[data.size() - 1];
    int N = data.size();
    
    double h = 2.0*IQR/pow(N, 1.0/3.0);
    std::cout << "    N = " << N << std::endl;
    std::cout << "    Histogram bin size: " << std::setprecision(15) << h << std::endl;
    
    int num_bins = ceil((max - min)/h);
    
    for (int i = 0; i < num_bins; ++i) {
        double val = min + (i + 0.5)*h;
        bins.push_back(val);
        hist.push_back(0);
    }
    
    for (int i = 0; i < N; ++i) {
        int bin = (data[i] - min)/h;
        hist[bin]++;
    }
}

#endif
