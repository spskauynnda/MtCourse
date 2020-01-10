#include "../tensor/XGlobal.h"
#include "../tensor/XTensor.h"
#include "../tensor/core/CHeader.h"

#ifndef _TESTNET_H__
#define _TESTNET_H__

using namespace nts;

namespace testnet {
     /* Global config for learning */
	
	struct Config {
		float initLearningRate;
		int nEpoch;
		float minmax;
		int h_size;
		int trainDataSize;
		int testDataSize;
		int devID;
	};
	// waited to be added:  [NarrowFactor]( RateDecreasing/gaps )

	
	/* Aimed model for a.s to true results */
	struct TestModel {
		/* Encoder-weights | Decoder-weights  */
		XTensor w1;
		XTensor w2;
		XTensor b;

		/* size of Hidden Layers(changeable for coming version...) */
		int h_size;
		int devID;
	};

	struct TestNet{
		/* name: h_{which_tensor_operated_with_me_before} */
		XTensor h_w1;
		XTensor h_b;
		XTensor h_activeFunc;
		XTensor h_w2;
	};
	int testNetMain(int argc, const char ** argv);
};

#endif