#include <iostream>
#include <armadillo>
#include <cmath>
#include <typeinfo>

using namespace std;
using namespace arma;

typedef std::vector<mat> vm;
typedef std::vector<double> vd;

enum Function
{
    SIGMOIDE,
    TANH,
    RELU
};

class MLP
{

private:
    int input;
    int output;
    vm weights;
    vm soutouts;

    mat ReLu(mat a)
    {
        mat::iterator it = a.begin();
        mat::iterator it_end = a.end();
        for (; it != it_end; ++it)
        {
            *it = max(0.0, *it);
        }
        return a;
    }

    mat softMax(mat a)
    {

        mat::iterator it = a.begin();
        mat::iterator it_end = a.end();
        double sum = 0;
        for (; it != it_end; ++it)
        {
            sum += exp(*it);
        }
        it = a.begin();
        for (; it != it_end; ++it)
        {
            *it = exp(*it) / sum;
        }
        return a;
    }
    mat tanh(mat a)
    {
        mat::iterator it = a.begin();
        mat::iterator it_end = a.end();
        for (; it != it_end; ++it)
        {
            *it = (exp(*it) - exp(-(*it))) / (exp(*it) + exp(-(*it)));
        }
        return a;
    }

    mat sigmoide(mat a)
    {
        mat::iterator it = a.begin();
        mat::iterator it_end = a.end();
        for (; it != it_end; ++it)
        {
            *it = 1 / (1 + exp(-*it));
        }
        return a;
    }

    mat forward(mat X)
    {
        mat A = X;
        soutouts.clear();
        for (unsigned i = 0; i < weights.size(); i++)
        {
            mat Z = A * weights[i];
            // cout << "Z = " << Z << endl;
            A = ReLu(Z);
            // cout << "tanh = " << A << endl;
            soutouts.push_back(A);
        }
        mat AF = softMax(A);
        soutouts.push_back(AF);
        return AF;
    }

    mat one_hot(mat Y)
    {
        mat one_hot_Y;
        one_hot_Y.zeros(1, output);
        double value = Y(0, 0);
        one_hot_Y(0, value) = 1;
        return one_hot_Y;
    }

    int predict_label(mat res)
    {
        return std::distance(res.begin(), std::max_element(res.begin(), res.end()));
    }

    mat dh0(mat delta, mat Xi)
    {
        // cout<<"delta = "<<delta<<endl;
        // cout<<"Xi = "<<Xi<<endl;
        return Xi.t() * delta;
    }

    mat dhk(mat &delta, mat W, mat Xi, mat Xj)
    {
        delta = delta * W.t();
        const int ncosl = delta.n_cols;
        for (int j = 0; j < ncosl; j++)
        {
            delta(0, j) *= Xj(0, j) * (1 - Xj(0, j));
        }
        return Xi.t() * delta;
    }

    vm backward(mat X, mat Y)
    {
        // vm dWs;
        auto one_hot_y = one_hot(Y);
        // cout << "one_hot_y = " << one_hot_y << endl;
        mat dZ = soutouts.back() - one_hot_y; // deltas
        // cout << "dZ = " << dZ << endl;
        // cout<<dZ<<endl;
        vm dWs;
        /*

              Layers
              ---W---
              i-----j               i = salida anterior  j = salida actual

              *el indice de cada weigth es el indice de la salida actual.

              hi ----- hk-1 ----- hk -------h0

              hi- input,hk - capa hiden k, hk-1 capa hiden k-1

              hk-ho
              mat Xi = soutouts[i-1];
              mat dW = Xi.t()*dZ;
              dWs.push_back(dW);

          */
        /*
             hk-1 -hk
             ---------
             Xi = soutouts[i-1]
             w = weights[i]
             Xj = soutouts[i]
             dZ=dZ*weights[i+1].t()
             for(int j =0; j<dZ.n_cols; i++){
                 dZ(0,j) *= Xj(0,j)*(1-Xj(0,j));
             }
             mat dW = Xi.t()*dZ;
             dWs.push_back(dW);


        */
        /*  hi - hk-1
            Xi = inputs
            Xj = soutouts[i]
            dZ=dZ*weights[i+1].t()
            for(int j =0; j<dZ.n_cols; i++){
                dZ(0,j) *= Xj(0,j)*(1-Xj(0,j));
            }
            mat dW = Xi.t()*dZ;
            dWs.push_back(dW);


        */
        for (int i = weights.size() - 1; i >= 0; i--)
        {

            if (i == weights.size() - 1)
            {
                mat dW = dh0(dZ, soutouts[i - 1]);
                dWs.push_back(dW);
            }
            else
            {
                mat Xi = i == 0 ? X : soutouts[i - 1];
                mat Xj = soutouts[i];
                mat dW = dhk(dZ, weights[i + 1], Xi, Xj);
                dWs.push_back(dW);
            }
        }
        std::reverse(dWs.begin(), dWs.end());
        return dWs;
    }
    void printMatrix(vm item)
    {
        for (auto i : item)
        {
            cout << i << endl;
        }
    }

    double CEE(mat Y, mat Y_hat)
    {
        // 1 0 0
        // 0.9 0.1 0.2
        //  double sum = 0;
        //  for (int i = 0; i < Y.n_cols; i++)
        //  {
        //      sum += -Y(0, i) * log(Y_hat(0, i));
        //  }
        //  return sum;

        double error = 0;
        for (int i = 0; i < Y.n_cols; ++i)
        {
            error = -Y(0, 1) * log(Y_hat(0, i)) - (1 - Y(0, 1)) * log(1 - Y_hat(0, i));
        }
        return error;
    }

public:
    MLP(int input, vd nodosh, int output, Function func = RELU)
    {

        this->input = input;
        this->output = output;

        mat init = randu(input, nodosh[0]);
        weights.push_back(init);

        for (unsigned i = 1; i < nodosh.size(); i++)
        {
            mat W = randu(nodosh[i - 1], nodosh[i]);
            weights.push_back(W);
        }
        mat W = randu(nodosh[nodosh.size() - 1], output);
        weights.push_back(W);
    }

    void update_params(vm dWs, double lr)
    {
        for (unsigned i = 0; i < dWs.size(); i++)
        {
            weights[i] -= dWs[i] * lr;
        }
    }

    template <typename T>
    void trainning(double epoch, double alpha, T X, T Y)
    {
        // auto rng = std::default_random_engine{};
        // std::vector<int> arr(Y.size());
        // iota(arr.begin(), arr.end(), 0);

        for (int i = 0; i < epoch; i++)
        {
            double errorF = 0.0;
          
            // std::shuffle(std::begin(arr), std::end(arr), rng);
            for (unsigned j = 0; j < Y.size(); j++)
            {
                int rIndex = j;//arr[j];
                mat x = X.row(rIndex);
                mat y = Y.row(rIndex);
                // cout << "x = " << x << ",y = " << y << endl;
                auto res = forward(x);
                // cout << "res = " << res << endl;
                // cout << "soutouts = " << endl;
                // printMatrix(soutouts);
                auto errorM = CEE(one_hot(y), res);
                errorF += errorM;
                // cout<<errorM<<endl;
                vm NdWs = backward(x, y);
                update_params(NdWs, alpha);
            }
            cout << errorF /Y.size() << endl;
            // cout << "Weigth finish " << i << endl;
            // printMatrix(weights);
        }
    }
};