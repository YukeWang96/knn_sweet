
#include<iomanip>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<vector>
#include<algorithm>
// #include<float.h>
#include<math.h>

#include <cuda_fp16.h>
#include "cublas_v2.h"

#define debug 0
using namespace std;

__half *sources;
__half *queries;
__half *qreps;
__half *sreps;

int source_nb = 0;
int query_nb = 0;
int qrep_nb = 0;
int srep_nb = 0;
int dim = 0;
int K = 0;


inline __host__ __device__ __half * getPoint(__half *A, int i){
	return A+i*dim;
}


typedef struct Point2Rep{
	int repIndex;
	__half dist2rep;
}P2R;

typedef struct IndexAndDist{
	int index;
	__half dist;
}IndexDist;

typedef struct repPoint_static{
	__half maxdist = 0.0f;
	__half mindist = FLT_MAX;
	uint npoints = 0;
	uint noreps = 0;
	__half kuboundMax = 0.0f;
}R2all_static;

typedef struct repPoint_static_dev{
	__half maxdist;
	__half mindist;
	uint npoints;
	uint noreps;
	__half kuboundMax;
}R2all_static_dev;

typedef struct repPoint_dynamic_v{
	vector<__half> Vquerymembers;
	vector<IndexDist> Vsortedmembers;
	vector<int> Vreplist;
}R2all_dyn_v;

typedef struct repPoint_dynamic_p{
	int *memberID;
	IndexDist *sortedmembers;
	__half *kubound;
	IndexDist *replist;
}R2all_dyn_p;


// struct sort_dec{
// 	bool operator()(const IndexDist &left, const IndexDist &right){
// 		return __hgt(left.dist, right.dist);
// 	}
// };

struct sort_inc{
	bool operator()(const IndexDist &left, const IndexDist &right){
		return (float)left.dist < (float)right.dist;
	}
};

// bool sort_inc(const IndexDist &left, const IndexDist &right){
// 	return left.dist < right.dist; 
// }

struct timespec t1,t2;
void timePoint(struct timespec &T1){
	clock_gettime(CLOCK_REALTIME, &T1);
}

__half timeLen(struct timespec &T1, struct timespec &T2){
	return T2.tv_sec-T1.tv_sec+(T2.tv_nsec-T1.tv_nsec)/1.e9;
}

void pointSetup(char *query_data, char *source_data){

    srand(2015);
if(strcmp((const char*)query_data, "random")==0){
	cout<<"query data random"<<endl;
	for(int i = 0; i < query_nb; i++){
		for(int j = 0; j < dim; j++){
			queries[i * dim + j] = __float2half(rand()%10 + (float)rand()/RAND_MAX);
		}
	}
}
else{
	FILE *fq = fopen(query_data,"r");
	if(fq == NULL){
		cout << "error opening query files"<<endl;
		exit(1);
	}
	for(int i = 0; i < query_nb; i++)
		for(int j = 0; j < dim; j++)
			if(fscanf(fq,"%f", &queries[i * dim + j])!=1)
				printf("error reading\n");
	fclose(fq);
}

if(strcmp((const char*)source_data, "random")==0){
	cout<<"source data random"<<endl;
	for(int i = 0; i < source_nb; i++){
		for(int j = 0; j < dim; j++){
			sources[i * dim + j] = __float2half(rand()%10 + (float)rand()/RAND_MAX);
		}
	}
}
else{
	FILE *fs = fopen(source_data,"r");
	if(fs == NULL){
		cout << "error opening query files"<<endl;
		exit(1);
	}
	for(int i = 0; i < source_nb; i++)
		for(int j = 0; j < dim; j++)
			fscanf(fs,"%f", &sources[i * dim + j]);
	fclose(fs);
}
}

typedef struct Float128{
        __half a[28];
}floatdim;
/*
__device__ __half Edistance_128(__half *a, __half *b, int dim=dim){
        __half distance = 0.0f;
		__half tmpA[28], tmpB[28];
		memcpy(tmpA, a, 28*sizeof(__half));
		memcpy(tmpB, b, 28*sizeof(__half));
        for(int i = 0; i < 28; i++){
			__half temp = tmpA[i] - tmpB[i];
			distance += temp * temp;
		}
        return hsqrt(distance);
}*/

__device__ __half Edistance_128(__half *a, __half *b, int dim=dim){
        __half distance = 0.0f;
		float4 *A = (float4 *)a;
		float4 *B = (float4 *)b;
		__half tmp = 0.0f;
        for(int i = 0; i < int(dim/4); i++){
                float4 a_local = A[i];
                float4 b_local = __ldg(&B[i]);
                        tmp = a_local.x- b_local.x;
                        distance += tmp * tmp;
                        tmp = a_local.y - b_local.y;
                        distance += tmp * tmp;
                        tmp = a_local.z - b_local.z;
                        distance += tmp * tmp;
                        tmp = a_local.w - b_local.w;
                        distance += tmp * tmp;
                }
		for(int i = int(dim/4)*4; i < dim; i++){
			tmp = (a[i])-(b[i]);
			distance += tmp * tmp;
        }
        return hsqrt(distance);
}

/*
__device__ __half Edistance_128(__half *a, __half *b, int dim=dim){
        __half distance = 0.0f;
		double4 *A = (double4 *)a;
		double4 *B = (double4 *)b;
		__half tmp = 0.0f;
        for(int i = 0; i < dim/8; i++){
                double4 a_local = A[i];
                double4 b_local = (B[i]);
				__half *a_l = (__half *)&a_local;
				__half *b_l = (__half *)&b_local;
                        tmp = a_l[0]- b_l[0];
                        distance += tmp * tmp;
                        tmp = a_l[1]- b_l[1];
                        distance += tmp * tmp;
                        tmp = a_l[2]- b_l[2];
                        distance += tmp * tmp;
                        tmp = a_l[3]- b_l[3];
                        distance += tmp * tmp;
                        tmp = a_l[4]- b_l[4];
                        distance += tmp * tmp;
                        tmp = a_l[5]- b_l[5];
                        distance += tmp * tmp;
                        tmp = a_l[6]- b_l[6];
                        distance += tmp * tmp;
                        tmp = a_l[7]- b_l[7];
                        distance += tmp * tmp;
                }
		for(int i = int(dim/8)*8; i < dim; i++){
			tmp = (a[i])-(b[i]);
			distance += tmp * tmp;
        }
        return hsqrt(distance);
}
*/
__device__ __half Edistance_gpu(__half *A, __half *B, int dim=dim){
	__half distance = 0.0f;
	for(int i = 0; i < dim; i++){
		__half tmp = __ldg(&A[i]) - __ldg(&B[i]);
		distance += tmp * tmp;
	}
	return hsqrt(distance);
}
__device__ __half Edistance(__half *A, __half *B, int dim=dim){
	__half distance = 0.0f;
	for(int i = 0; i < dim; i++){
		__half tmp = A[i]- B[i];
		distance += tmp * tmp;
	}
	return hsqrt(distance);
}
// void selectReps(__half *points, int point_nb, __half *reps,int rep_nb){
// 	int totalTest = 10;
// 	vector<int> myrandom;
// 	for(int i = 0; i < point_nb; i++)
// 		myrandom.push_back(i);

// 	//random_shuffle(myrandom.begin(), myrandom.end());
// 	int *qIndex = (int *)malloc(totalTest * rep_nb * sizeof(int));
// 	__half bestdistance = -1.0f;
// 	int bestTest = -1;
// 	if(rep_nb > point_nb) rep_nb = point_nb;


// 	srand(2015);
// 	for(int test = 0; test < totalTest; test++){
// 		//random_shuffle(myrandom.begin(), myrandom.end());
// 		for(int i = 0; i < rep_nb; i++){
// 			qIndex[rep_nb * test + i] = rand()%point_nb;//myrandom[i];
// 		}

// 		__half distance = 0.0f;
// 		for(int i = 0; i < rep_nb; i++){
// 			for(int j = 0; j < i; j++){
// 				distance += Edistance(getPoint(points,qIndex[rep_nb * test + i]), getPoint(points,qIndex[rep_nb * test + j]));
// 			}

// #if debug
// 			if(test==3)
// 			for(int kk = 0; kk < dim; kk++){
// 				cout<< "i="<<i<<" index="<<qIndex[rep_nb * test + i] <<" queries["<<kk<<"]= "<<queries[qIndex[rep_nb * test + i]*dim+kk]<<endl;

// 			}
// #endif
// 		}

// 		if(bestdistance < distance){
// 			bestdistance = distance;
// 			bestTest = test;
// 			// cout<<"test = "<<test<<" distance = "<<distance<<endl;
// 		}
// 	}

// 	cout<<"bestTest = "<<bestTest << endl;
// 	for(int i = 0; i < rep_nb; i++){
// 		memcpy((void *)&reps[i * dim], (const void *)&points[qIndex[rep_nb * bestTest + i]*dim], dim * sizeof(__half));
// 	}

// #if debug
// 		for(int i = 0; i < rep_nb; i++){
// 			for(int kk = 0; kk < dim; kk++){
// 				cout<< "i="<<i<<" reps[ "<<kk<<"] = "<<fixed<<std::setprecision(10)<<reps[i*dim +kk]<<endl;
// 			}
// 		}
// #endif

// 	free(qIndex);

// }
