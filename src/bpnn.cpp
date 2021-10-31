/*
 * Translation of python Back-Propagation Neural Network
 * http://arctrix.com/nas/python/bpnn.py
 */
 
#include <math.h>
#include <EEPROM.h>

#include "bpnn.h"
#include "Arduino.h" 

float BPNN::randValue(float min, float max)
{
    return (float(random(1000)) / 1000) * (max - min) + min;
}

float BPNN::sigmoid(float x)
{
    return tanh(x);
}

float BPNN::dsigmoid(float y)
{
    return 1 - y * y;
}

float* BPNN::makeVector(byte length, float fill)
{
    float* result = new float[length];
    for (byte i = 0; i< length; i++)
    {
        result[i] = fill;
    }
    return result;
}

float** BPNN::makeMatrix(byte rows, byte cols, float fill)
{
    float** result = new float*[rows];
    for (byte i = 0; i< rows; i++)
    {
        result[i] = new float[cols];
        for (byte j = 0; j < cols; j++)
        {
            result[i][j] = fill;
        }
    }

    return result;
}

float BPNN::backpropagate(float* targets, float N, float M)
{
    float* output_deltas = this->makeVector(this->no, 0);
    float error;

    // calculate error terms for output
    for (byte k = 0; k < this->no; k++)
    {
        error = targets[k] - this->ao[k];
        output_deltas[k] = this->dsigmoid(this->ao[k]) * error;
    }

    // calculate error terms for hidden
    float* hidden_deltas = this->makeVector(this->nh, 0);
    for (byte j = 0; j < this->nh; j++)
    {
        error = 0;
        for (byte k = 0; k < this->no; k++)
        {
            error += output_deltas[k] * this->wo[j][k];
        }
        hidden_deltas[j] = this->dsigmoid(this->ah[j]) * error;
    }

    float change;
    // update output weights
    for (byte j = 0; j < this->nh; j++)
    {
        for (byte k = 0; k < this->no; k++)
        {
            change = output_deltas[k] * this->ah[j];
            this->wo[j][k] += N * change + M * this->co[j][k];
            this->co[j][k] = change;
        }
    }

    // update input weights
    for (byte i = 0; i < this->ni; i++)
    {
        for (byte j = 0; j < this->nh; j++)
        {
            change = hidden_deltas[j] * this->ai[i];
            this->wi[i][j] += N * change + M * this->ci[i][j];
            this->ci[i][j] = change;
        }
    }

    delete[] output_deltas;
    delete[] hidden_deltas;

    // calculate error
    error = 0;
    for (byte k = 0; k < this->no; k++)
    {
        error += 0.5 * (targets[k] - this->ao[k]) * (targets[k] - this->ao[k]);
    }

    return error;
}

BPNN::BPNN(byte inputs, byte hidden, byte outputs)
{
    this->ni = inputs + 1; // +1 for bias node
    this->nh = hidden;
    this->no = outputs;

    this->ai = this->makeVector(this->ni, 1.0);
    this->ah = this->makeVector(this->nh, 1.0);
    this->ao = this->makeVector(this->no, 1.0);

    this->wi = this->makeMatrix(this->ni, this->nh, 0.0);
    this->wo = this->makeMatrix(this->nh, this->no, 0.0);

    srand(millis());
    
    for (byte i = 0; i < this->ni; i++)
    {
        for (byte j = 0; j < this->nh; j++)
        {
            this->wi[i][j] = this->randValue(-0.2, 0.2);
        }
    }

    for (byte j = 0; j < this->nh; j++)
    {
        for (byte k = 0; k < this->no; k++)
        {
            this->wo[j][k] = this->randValue(-2.0, 2.0);
        }
    }

    this->ci = this->makeMatrix(this->ni, this->nh, 0.0);
    this->co = this->makeMatrix(this->nh, this->no, 0.0);    
}


float* BPNN::compute(float* inputs)
{
    float sum = 0;
    
    for (byte i = 0; i < this->ni - 1; i++)
    {
        this->ai[i] = inputs[i];         
    }

    for (byte j = 0; j < this->nh; j++)
    {
        sum = 0;
        for (byte i = 0; i < this->ni; i++)
        {
            sum += this->ai[i] * this->wi[i][j];
            this->ah[j] = this->sigmoid(sum);    
        }
    }

    for (byte k = 0; k < this->no; k++)
    {
        sum = 0;
        for (byte j = 0; j < this->nh; j++)
        {
            sum += this->ah[j] * this->wo[j][k];
            this->ao[k] = this->sigmoid(sum);    
        }
    }

    return this->ao;
}

void BPNN::train(float** patterns, float** targets, int numpatterns, int iterations, float desired_error, float learningrate, float momentum)
{
    float error;
    float* pattern;
    float* target;
    
    for (int i = 1; i <= iterations; i++)
    {
        error = 0;            
        
        for (int j = 0; j < numpatterns; j++)
        {
            pattern = patterns[j];
            target = targets[j];
            this->compute(pattern);
            error += this->backpropagate(target, learningrate, momentum);                                    
        }        
        
        if (error < desired_error)
        {
            break;    
        }
        
    }
}

void BPNN::load()
{
    int address = 0;
    // save weights
    for (byte i = 0; i < this->ni; i++)
    {
        for (byte j = 0; j < this->nh; j++)
        {
            EEPROM.get(address, this->wi[i][j]);
            address += sizeof(float);
        }
    }

    for (byte j = 0; j < this->nh; j++)
    {
        for (byte k = 0; k < this->no; k++)
        {
            EEPROM.get(address, this->wo[j][k]);
            address += sizeof(float);
        }
    }        
}


void BPNN::save()
{
    int address = 0;
    // save weights
    for (byte i = 0; i < this->ni; i++)
    {
        for (byte j = 0; j < this->nh; j++)
        {
            EEPROM.put(address, this->wi[i][j]);
            address += sizeof(float);
        }
    }

    for (byte j = 0; j < this->nh; j++)
    {
        for (byte k = 0; k < this->no; k++)
        {
            EEPROM.put(address, this->wo[j][k]);
            address += sizeof(float);
        }
    }        
}

