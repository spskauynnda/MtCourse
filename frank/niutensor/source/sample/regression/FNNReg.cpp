#include "FNNReg.h"
#include "../../tensor/function/FHeader.h"
namespace fnnreg
{
/*base parameter*/
float learningRate = 0.3F;           // learning rate
int nEpoch = 100;                      // max training epochs
float minmax = 0.01F;                 // range [-p,p] for parameter initialization

void Init(FNNRegModel &model);
void InitGrad(FNNRegModel &model, FNNRegModel &grad);
void Train(float *trainDataX, float *trainDataY, int dataSize, FNNRegModel &model);
void Forword(XTensor &input, FNNRegModel &model, FNNRegNet &net);
void MSELoss(XTensor &output, XTensor &gold, XTensor &loss);
void Backward(XTensor &input, XTensor &gold, FNNRegModel &model, FNNRegModel &grad, FNNRegNet &net);
void Update(FNNRegModel &model, FNNRegModel &grad, float learningRate);
void CleanGrad(FNNRegModel &grad);
void Test(float *testData, int testDataSize, FNNRegModel &model);

int FNNRegMain(int argc, const char ** argv)
{
    FNNRegModel model;
    model.h_size = 4;
    const int dataSize = 16;
    const int testDataSize = 3;
    model.devID = 0;
    Init(model);

    /*train Data*/
    float trainDataX[dataSize] = { 51,56.8,58,63,66,69,73,76,81,85,90,94,97,100,103,107 };
    float trainDataY[dataSize] = { 31,34.7,35.6,36.7,39.5,42,42.7,47,49,51,52.5,54,55.7,56,58.8,59.2 };

    float testDataX[testDataSize] = { 64, 80, 95 };

    Train(trainDataX, trainDataY, dataSize, model);

    Test(testDataX, testDataSize, model);
    return 0;
}

void Init(FNNRegModel &model)
{
    InitTensor2D(&model.weight1, 1, model.h_size, X_FLOAT, model.devID);
    InitTensor2D(&model.weight2, model.h_size, 1, X_FLOAT, model.devID);
    InitTensor2D(&model.b, model.h_size, 1, X_FLOAT, model.devID);
    model.weight1.SetDataRand(-minmax, minmax);
    model.weight2.SetDataRand(-minmax, minmax);
    model.b.SetZeroAll();
    printf("Init model finish!\n");
}

void InitGrad(FNNRegModel &model, FNNRegModel &grad)
{
    InitTensor(&grad.weight1, &model.weight1);
    InitTensor(&grad.weight2, &model.weight2);
    InitTensor(&grad.b, &model.b);

    grad.h_size = model.h_size;
    grad.devID = model.devID;
}

void Train(float *trainDataX, float *trainDataY, int dataSize, FNNRegModel &model)
{
    printf("prepare data for train\n");
    /*prepare for train*/
    TensorList inputList;
    TensorList goldList;
    for (int i = 0; i < dataSize; ++i)
    {
        XTensor*  inputData = NewTensor2D(1, 1, X_FLOAT, model.devID);
        inputData->Set2D(trainDataX[i] / 100, 0, 0);
        inputList.Add(inputData);

        XTensor*  goldData = NewTensor2D(1, 1, X_FLOAT, model.devID);
        goldData->Set2D(trainDataY[i] / 60, 0, 0);
        goldList.Add(goldData);
    }

    /*check data*/
    /*
    for (int i = 0; i < 16; ++i)
    {
        XTensor* tmp = inputList.GetItem(i);
        tmp->Dump(stderr);

        tmp = goldList.GetItem(i);
        tmp->Dump(stderr);
    }
    */

    printf("start train\n");
    FNNRegNet net;
    FNNRegModel grad;
    InitGrad(model, grad);
    for (int epochIndex = 0; epochIndex < nEpoch; ++epochIndex)
    {
        printf("epoch %d\n",epochIndex);
        float totalLoss = 0;
        if ((epochIndex + 1) % 50 == 0)
            learningRate /= 3;
        for (int i = 0; i < inputList.count; ++i)
        {
            XTensor *input = inputList.GetItem(i);
            XTensor *gold = goldList.GetItem(i);
            Forword(*input, model, net);
            //output.Dump(stderr);
            XTensor loss;
            MSELoss(net.output, *gold, loss);
            //loss.Dump(stderr);
            totalLoss += loss.Get1D(0);

            Backward(*input, *gold, model, grad, net);

            Update(model, grad, learningRate);

            CleanGrad(grad);

        }
        printf("%f\n", totalLoss / inputList.count);
    }
}

void Forword(XTensor &input, FNNRegModel &model, FNNRegNet &net)
{
    net.hidden_state1 = MatrixMul(input, model.weight1);
    net.hidden_state2 = net.hidden_state1 + model.b;
    net.hidden_state3 = HardTanH(net.hidden_state2);
    net.output = MatrixMul(net.hidden_state3, model.weight2);
}

void MSELoss(XTensor &output, XTensor &gold, XTensor &loss)
{
    XTensor tmp = output - gold;
    loss = ReduceSum(tmp, 1, 2) / output.dimSize[1];
}

void MSELossBackword(XTensor &output, XTensor &gold, XTensor &grad)
{
    XTensor tmp = output - gold;
    grad = tmp * 2;
}

void Backward(XTensor &input, XTensor &gold, FNNRegModel &model, FNNRegModel &grad, FNNRegNet &net)
{
    XTensor lossGrad;
    XTensor &dedw2 = grad.weight2;
    XTensor &dedb = grad.b;
    XTensor &dedw1 = grad.weight1;
    MSELossBackword(net.output, gold, lossGrad);
    MatrixMul(net.hidden_state3, X_TRANS, lossGrad, X_NOTRANS, dedw2);
    XTensor dedy = MatrixMul(lossGrad, X_NOTRANS, model.weight2, X_TRANS);
    _HardTanHBackward(&net.hidden_state3, &net.hidden_state2, &dedy, &dedb);
    dedw1 = MatrixMul(input, X_NOTRANS, dedb, X_TRANS);
}

void Update(FNNRegModel &model, FNNRegModel &grad, float learningRate)
{
    model.weight1 = Sum(model.weight1, grad.weight1, -learningRate);
    model.weight2 = Sum(model.weight2, grad.weight2, -learningRate);
    model.b = Sum(model.b, grad.b, -learningRate);
}

void CleanGrad(FNNRegModel &grad)
{
    grad.b.SetZeroAll();
    grad.weight1.SetZeroAll();
    grad.weight2.SetZeroAll();
}

void Test(float *testData, int testDataSize, FNNRegModel &model)
{
    FNNRegNet net;
    XTensor*  inputData = NewTensor2D(1, 1, X_FLOAT, model.devID);
    for (int i = 0; i < testDataSize; ++i)
    {

        inputData->Set2D(testData[i] / 100, 0, 0);

        Forword(*inputData, model, net);
        float ans = net.output.Get2D(0, 0) * 60;
        printf("%f\n", ans);
    }

}

};