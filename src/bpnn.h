/*
 * Translation of python Back-Propagation Neural Network
 * http://arctrix.com/nas/python/bpnn.py
 */
#ifndef BPNN_H
#define BPNN_H

#include <arduino.h>
#include <math.h>

class BPNN
{
    private:
        byte ni;
        byte nh;
        byte no;

        float* ai;
        float* ah;
        float* ao;

        float** wi;
        float ** wo; 

        float** ci;
        float** co;

        float randValue(float min, float max);
        float sigmoid(float x);
        float dsigmoid(float y);
        float* makeVector(byte length, float fill);
        float** makeMatrix(byte rows, byte cols, float fill);
        float backpropagate(float* targets, float N, float M);
    public: 
        BPNN(byte inputs, byte hidden, byte outputs);
        
        float* compute(float* inputs);        
        void train(float** patterns, float** targets, int numpatterns, int iterations, float desired_error, float learningrate, float momentum);

        void load();
        void save();
};
#endif



