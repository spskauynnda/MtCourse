#include "testNet.h"
#include "../tensor/function/FHeader.h"

namespace testnet {
	/* global constant (waited for listed to be set) */
	/* learning rate */
	/* max training Epoch */

	#pragma once 
	Config testConfig = { 
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
	void Train(TestModel &model, TestModel &grad, Config testConfig, TensorList xList, TensorList yList);
		void Forward(TestNet &hidNet, TestModel model, XTensor input);
		void MSELoss(XTensor &loss, XTensor output, XTensor result);
		void Backward(TestModel &grad, TestModel model, TestNet hidNet, XTensor input, XTensor result);
		void Update(TestModel &model, TestModel grad, float learningRate);
		void CleanGrad(TestModel &grad);
	void Test(float* testData, Config testConfig, TestModel model);
	

	int testNetMain(int argc, const char ** argv) {		
		TestModel model;
		TestModel grad;
		TensorList xList;
		TensorList yList;
		Init(model, grad, testConfig, xList, yList, trainDataX, trainDataY);
		Train(model, grad, testConfig, xList, yList);
		Test(testDataX, testConfig, model);
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

	void Train(TestModel& model, TestModel &grad, Config testConfig, TensorList xList, TensorList yList) {
		TestNet hidNet;
		for (int i = 0; i < testConfig.nEpoch; i++) {
			printf("Loop: %d\n", i);
			float totalLoss = 0;
			if (i % 50 == 0) {
				learningRate *= 0.4;
			}
			
			for (int j = 0; j < testConfig.trainDataSize; j++) {
				printf("innerj:%d  |", j); //
				XTensor *pXi = xList.GetItem(j);
				XTensor *pYi = yList.GetItem(j);
				printf("  forward |"); //
				Forward(hidNet, model, *pXi);
				printf("  loss  |"); // 
				XTensor loss;
				MSELoss(loss, hidNet.h_w2, *pYi);
				printf("  sumloss  |"); // 
				totalLoss += loss.Get1D(0);
				printf("  grad  |"); // 
				Backward(grad, model, hidNet, *pXi, *pYi);
				printf("  update  |"); // 
				Update(model, grad, learningRate);
				CleanGrad(grad);
			}
		}

	}

	void Forward(TestNet& hidNet, TestModel model, XTensor input) {
		hidNet.h_w1 = MatrixMul(input, model.w1);
		hidNet.h_b = hidNet.h_w1 + model.b;
		hidNet.h_activeFunc = HardTanH(hidNet.h_b);
		hidNet.h_w2 = MatrixMul(model.w2, hidNet.h_activeFunc);
	}

	void MSELoss(XTensor& loss, XTensor output, XTensor result) {
		XTensor tmp = output - result;
		// shift = 2 ??
		//loss = ReduceSum(tmp, 1, 2) / output.dimSize[1];
		loss = ReduceSum(tmp, 1, 2);
	}

	void Backward(TestModel &grad, TestModel model, TestNet hidNet, XTensor input, XTensor result) {
		// completely copied by Sample
		XTensor lossGrad;
		XTensor &dedw2 = grad.w2;
		XTensor &dedb = grad.b;
		XTensor &dedw1 = grad.w1;
		
		// MESLossGrad
		XTensor tmp = hidNet.h_w2 - result;
		lossGrad = tmp * 2;

		MatrixMul(hidNet.h_activeFunc, X_TRANS, lossGrad, X_NOTRANS, dedw2);
		XTensor dedy = MatrixMul(lossGrad, X_NOTRANS, model.w2, X_TRANS);
		_HardTanHBackward(&hidNet.h_activeFunc, &hidNet.h_b, &dedy, &dedb);
		dedw1 = MatrixMul(input, X_NOTRANS, dedb, X_TRANS);
	}

	void Update(TestModel &model, TestModel grad, float learningRate) {
		model.w1 = Sum(model.w1, grad.w1, -learningRate);
		model.w2 = Sum(model.w2, grad.w2, -learningRate);
		model.b = Sum(model.b, grad.b, -learningRate);
	}

	void CleanGrad(TestModel& grad) {
		grad.b.SetZeroAll();
		grad.w1.SetZeroAll();
		grad.w2.SetZeroAll();
	}

	void Test(float *testData, Config testConfig, TestModel model) {
		TestNet hidNet;
		XTensor *inputData = NewTensor2D(1, 1, X_FLOAT, model.devID);
		for (int i = 0; i < testConfig.testDataSize; i++) {
			inputData->Set2D(testData[i] / 100 , 0, 0);
			Forward(hidNet, model, *inputData);
			float ans = hidNet.h_w2.Get2D(0, 0) * 60;
			printf("%f\n", ans);
		}
	}
	
};
