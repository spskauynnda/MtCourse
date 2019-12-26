#include "../tensor/XGlobal.h"
#include "../tensor/XTensor.h"
#include "../tensor/core/CHeader.h"

#ifndef _XORNET_H__
#define _XORNET_H__

using namespace nts;

namespace xornet {
     /* Global config for learning */
	
	struct Config {
		float initLearningRate;
		int nEpoch;
		float minmax;
		int h_size;
		int inputNum; // number of input variables
		int trainDataSize;
		int testDataSize;
		int devID;
	};
	
	/* Aimed model for a.s to true results */
	struct XorModel {
		/* Encoder-weights | Decoder-weights  */
		XTensor w1;
		XTensor w2;
		XTensor b;

		/* size of Hidden Layers(changeable for coming version...) */
		int h_size;
		int devID;
	};

	struct XorNet{
		/* name: h_{which_tensor_operated_with_me_before} */
		XTensor h_w1;
		XTensor h_b;
		XTensor h_activeFunc;
		XTensor h_w2;
	};
	int XorNetMain(int argc, const char ** argv);
};

#endif