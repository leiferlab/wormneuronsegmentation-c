#include <opencv2/core.hpp>

void z_gaussian_blur(uint16_t VolIn[], float VolOut[], 
	int32_t sizez, int32_t size_x, int32_t sizey) {
	
	int size_xy = size_x*size_y;
	float val = 0.0;
	float sum = 0.0;
	
	for(int z=1;z<sizez-1;z++){
	    for(int i=z*size_xy;i<(z+1)*size_xy;i++){
	        sum = 0.0;
	        for(int q=-1;q<2;q++) {
	            val = (float) VolIn[(z+q)*size_xy+x*size_x+y];
	            sum = sum + coeff[q+1]*val;
	        }
	        VolOut[z*size_xy+x*size_x+y] = sum;
	    }
	}
}
