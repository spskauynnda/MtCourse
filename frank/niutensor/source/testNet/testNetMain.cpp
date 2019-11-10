#include "testNet.h"
#include "../tensor/function/FHeader.h"

//#include "../tensor/function/FHeader.h"

namespace testnet {
	/* global constant (waited for listed to be set) */
	/* learning rate */
	/* max training Epoch */
	struct Config testConfig = { 
		0.3F,   // initLearningRate
		100,    // nEpoch
		0.01F,  // minmax
		4,      // h_size
		16,     // trainDataSize
		3,      // testDataSize
		0       // devID
	};
	
	float trainDataX[] = { 51, 56.8, 58, 63, 66, 69, 73, 76, 81, 85, 90, 94, 97, 100, 103,107 };
	float trainDataY[] = { 31, 34.7, 35.6, 36.7, 39.5, 42, 42.7, 47, 49, 51, 52.5, 54, 55.7, 56, 58.8, 59.2 };
	float testDataX[]  = { 64, 80, 95 };

	float learningRate = testConfig.initLearningRate;

	void Init(TestModel &model, TestModel &grad, Config testConfig, TensorList& xList, TensorList& yList, float* trainDataX, float* trainDataY);
	void Train(TestModel &model, TestModel grad, Config testConfig, TensorList xList, TensorList yList);
		void Forward(TestNet &hidNet, TestModel model, XTensor input);
	void Test(TestModel model, Config testConfig);
	void MSELoss(XTensor& loss, XTensor h_w2, XTensor y);

	int testNetMain(int argc, const char ** argv) {		
		TestModel model;
		TestModel grad;
		TensorList xList;
		TensorList yList;
		Init(model, grad, testConfig, xList, yList, trainDataX, trainDataY);
		Train(model, grad, testConfig, xList, yList);
		Test(model, testConfig);
		return 0;
	}

	void Init(TestModel& model, TestModel& grad, Config testConfig, TensorList& xList, TensorList& yList, float* trainDataX, float* trainDataY) {
		/* Model initialization */
		model.h_size = testConfig.h_size;
		model.devID = testConfig.devID;
		InitTensor2D(&model.w1, 1, model.h_size, X_FLOAT, model.devID);
		InitTensor2D(&model.w2, model.h_size, 1, X_FLOAT, model.devID);
		InitTensor2D(&model.b, model.h_size, 1, X_FLOAT, model.devID);
		model.w1.SetDataRand(-testConfig.minmax, testConfig.minmax);
		model.w2.SetDataRand(-testConfig.minmax, testConfig.minmax);
		model.b.SetZeroAll();
		printf("Model initialization completed\n");

		/* Grad initialization */
		InitTensor(&grad.w1, &model.w1);
		InitTensor(&grad.w2, &model.w2);
		InitTensor(&grad.b, &model.b);
		grad.h_size = model.h_size;
		grad.devID = model.devID;
		printf("Grad initialization completed\n");

		/* Data initialization  */
		for (int i = 0; i < testConfig.testDataSize; i++) {
			XTensor* pXi = NewTensor2D(1, 1, X_FLOAT, model.devID);
			pXi->Set2D(trainDataX[i], 0, 0);
			xList.Add(pXi);
			
			XTensor* pYi = NewTensor2D(1, 1, X_FLOAT, model.devID);
			pYi->Set2D(trainDataY[i], 0, 0);
			yList.Add(pYi);
		}
		printf("Data initialization completed\n");
	}

	void Train(TestModel& model, TestModel grad, Config testConfig, TensorList xList, TensorList yList) {
		TestNet hidNet;
		for (int i = 0; i < testConfig.nEpoch; i++) {
			printf("Loop: %d\n", i);
			float totalLoss = 0;
			for (int j = 0; j < testConfig.trainDataSize; j++) {
				XTensor* pXi = xList.GetItem(j);
				XTensor* pYi = xList.GetItem(j);
				Forward(hidNet, model, *pXi);

				XTensor loss; // for expansion
				MSELoss(hidNet.h_w2, *pYi, loss);
				totalLoss += loss.Get1D(0);
				
				Backward(XTensor );

				Update();
			}
		}

	}

	void Forward(TestNet& hidNet, TestModel model, XTensor input) {
		hidNet.h_w1 = MatrixMul(input, model.w1);
		hidNet.h_b = hidNet.h_w1 + model.b;
		hidNet.h_activeFunc = HardTanH(hidNet.h_b);
		hidNet.h_w2 = MatrixMul(model.w2, hidNet.h_activeFunc);
	}

	void MSELoss(XTensor& loss, XTensor h_w2, XTensor y) {
		loss = ReduceSum(y - h_w2, 1, 2) / h_w2.dimSize[1];
	}

	void Test(TestModel model, Config testConfig) {

	}
	
};
