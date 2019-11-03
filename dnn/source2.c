#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <io.h>
#include <conio.h>
#include <math.h>
#include <time.h>

#include "dnn_hmm.h"
#include "dnn_w.h"
#define HEADER_NEXT "dnn_w1.h"

#ifndef N_LAYER
#define N_LAYER 3
#define N_HIDDEN 39
#define N_OUT 63
#endif

#define N_DIMENSION 39
#define MAX_TIME 600
#define CUT 5
#define WEIGHT 0.00001
#define INITIALIZE 0
#define ETA 0.0002

int pro_index(char *name);

int main(){
	srand(time(NULL));
	int l = 0,T = 0, t = 0, d = 0, i = 0, j = 0, k = 0;
	int tempi = 0;
	float sum = 0, tempf = 0;;
	float h[N_LAYER - 1][N_HIDDEN]; // input, hidden layer output
	float out[N_OUT]; // output layer output
	float h_delta[N_LAYER - 1][N_HIDDEN];
	float out_delta[N_OUT];

	// make list of trn
	char trn_left[1300][_MAX_PATH];
	int trn_num = 0;
	FILE *trn_mono = fopen("trn_mono_state.txt","r");
	fgets(trn_left[0], _MAX_PATH, trn_mono); //drop first line
	fgets(trn_left[0], _MAX_PATH, trn_mono);
	trn_left[0][strlen(trn_left[0]) - 1] = '\0';
//	printf("%s\n",trn_left[0]);
	trn_num++;
	while(fgets(trn_left[trn_num], _MAX_PATH, trn_mono)){
		trn_left[trn_num][strlen(trn_left[trn_num]) - 1] = '\0';
		if(!strcmp(trn_left[trn_num],".")) {
			fgets(trn_left[trn_num], _MAX_PATH, trn_mono);
			trn_left[trn_num][strlen(trn_left[trn_num]) - 1] = '\0';
//			printf("%s\n",trn_left[trn_num]);
			trn_num++;
		}
	}
//	printf("%d\n",trn_num);
	fclose(trn_mono);

	char trn[_MAX_PATH];
	for(;trn_num != 0; trn_num--){
		// choose random one
		int index = rand()%trn_num;
		strcpy(trn, trn_left[index]);
		
		trn_mono = fopen("trn_mono_state.txt","r");
		do{
			fgets(trn_left[1299], _MAX_PATH, trn_mono);
			trn_left[1299][strlen(trn_left[1299]) - 1] = '\0';			
		}while(strcmp(trn_left[1299],trn));
		
		for(i = 0; i < strlen(trn) - 5; i++){
			trn[i] = trn[i + 1];
		}
		trn[i] = '\0';
		strcat(trn,"txt");

		FILE *trn_file = fopen(trn,"r");		
		fscanf(trn_file,"%d %d", &T, &d);
//		printf("%d %d\n",T,d);

		float x[MAX_TIME][N_DIMENSION] = {};
		float r[MAX_TIME][N_OUT] = {};
		// get input xt, rt
		for(t = 0; t < T; t++){

			for(d = 0; d < N_DIMENSION; d++){
				fscanf(trn_file,"%e", &x[t][d]);
			}

			fscanf(trn_mono,"%s %d", trn_left[1299], &d);
			r[t][pro_index(trn_left[1299])*3 + d - 1] = 1;
//			printf("%s %d\n",trn_left[1299],d);
		}
		fclose(trn_file);
		fclose(trn_mono);
		
		for(T = T - 2 * CUT; T != 0; T--){
			// choose random one
			t = CUT + rand()%T;
			
			// input layer output
			for(i = 0; i < N_HIDDEN; i++){
				h[0][i] = in_w[i][0];
				for(d = 0; d < N_DIMENSION; d++){
					h[0][i] += (in_w[i][d+1] * x[t][d]);
				}
				h[0][i] = 1.0 / (1.0 + exp(h[0][i]));
			}
			// hidden layer output
			for(l = 0; l < N_LAYER - 2 ; l++){
				for(i = 0; i < N_HIDDEN; i++){
					h[l+1][i] = hidden_w[l][i][0];
					for(j = 0; j < N_HIDDEN; j++){
						h[l+1][i] += (hidden_w[l][i][j+1] * h[l][j]);
					}
					h[l+1][i] = 1.0 / (1.0 + exp(h[l+1][i]));
				}
			}
			// output layer output
			sum = 0;
			for(i = 0; i < N_OUT; i++){
				out[i] = out_w[i][0];
				for(j = 0; j < N_HIDDEN; j++){
					out[i] += (out_w[i][j+1] * h[N_LAYER - 2][j]);
				}
				sum += out[i];
			}
			for(i = 0; i < N_OUT; i++){
				out[i] = out[i] / sum;
			}
			
			// compute delta, update w
			for(i = 0; i < N_OUT; i++){
				out_delta[i] = r[t][i] - out[i];
			}
			l = N_LAYER - 1;
			for(j = 0; j < N_HIDDEN; j++){
				h_delta[l-1][j] = 0;
				for(i = 0; i < N_OUT; i++){
					h_delta[l-1][j] += (out_delta[i] * out_w[i][j+1] * out[j] * (1 - out[j]));
				}
			}
			for(i = 0; i < N_OUT; i++){
				for(j = 0; j < N_HIDDEN; j++){
					out_w[i][j+1] += (ETA * out_delta[i] * h[l-1][j]);
				}
			}
			
			for(l = N_LAYER - 2; l >= 0; l--){
				if(1 != 0){
					for(j = 0; j < N_HIDDEN; j++){
						h_delta[l-1][j] = 0;
						for(i = 0; i < N_HIDDEN; i++){
							h_delta[l-1][j] += (h_delta[l][i] * hidden_w[l][i][j+1] * h[l-1][j] * (1 - h[l-1][j]));
						}
					}
				}
				for(i = 0; i < N_HIDDEN; i++){
					if(l != 0) {
						for(j = 0; j < N_HIDDEN; j++){
							hidden_w[l-1][i][j+1] += (ETA * h_delta[l][i] * h[l-1][j]);
						}
					}
					else {
						for(j = 0; j < N_DIMENSION; j++){
							in_w[i][j+1] += (ETA * h_delta[l][i] * x[t][j]);
						}
					}
				}
			}
			
			// go to end after loop
			for(d = 0; d < N_DIMENSION; d++){
				tempf = x[t][d];
				x[t][d] = x[CUT + T-1][d];
				x[CUT + T-1][d] = tempf;
			}
			for(d = 0; d < N_OUT; d++){
				tempi = r[t][d];
				r[t][d] = r[CUT + T-1][d];
				r[CUT + T-1][d] = tempi;
			}
		}
		
		// go to end after loop
		strcpy(trn_left[1299],trn_left[index]); //use trn_left[1299] as temp.
		strcpy(trn_left[index],trn_left[trn_num-1]);
		strcpy(trn_left[trn_num-1],trn_left[1299]);
		printf("%s %d\n", trn, trn_num);
		if(INITIALIZE) break;
	}


	FILE *header = fopen(HEADER_NEXT,"w");
	fprintf(header,"#define N_LAYER		%d\n",N_LAYER);
	fprintf(header,"#define N_HIDDEN	%d\n",N_HIDDEN);
	fprintf(header,"#define N_OUT		%d\n\n",N_OUT);
	fprintf(header,"float in_w[N_HIDDEN][N_DIMENSION + 1] = {\n");
	for(l = 0; l < N_HIDDEN; l++){
		fprintf(header,"{ ");		
		for(d = 0; d < N_DIMENSION + 1; d++){
			if(INITIALIZE) {
				if(d == 0) fprintf(header,"%e ", 0.0);
				else fprintf(header,"%e ",WEIGHT * (rand()%100000));
			}
			else fprintf(header,"%e ",in_w[l][d]);
			if(d != N_DIMENSION) fprintf(header,",");
		}
		fprintf(header,"}");
		if(l != N_HIDDEN - 1) fprintf(header,",\n");
	}
	fprintf(header,"};\n\n");

	fprintf(header,"float hidden_w[N_LAYER - 2][N_HIDDEN][N_HIDDEN + 1] = {\n");	
	for(l = 0; l < N_LAYER - 2; l++){
		fprintf(header,"{");		
		for(i = 0; i < N_HIDDEN; i++){
			fprintf(header,"{ ");		
			for(j = 0; j < N_HIDDEN + 1; j++){
				if(INITIALIZE) {
					if(j == 0) fprintf(header,"%e ", 0.0);
					else fprintf(header,"%e ",WEIGHT * (rand()%100000));
				}
				else fprintf(header,"%e ",hidden_w[l][i][j]);
				if(j != N_HIDDEN) fprintf(header,",");
			}
			fprintf(header,"}");
			if(i != N_HIDDEN - 1) fprintf(header,",\n");
		}
		fprintf(header,"}");
		if(l != N_LAYER - 3) fprintf(header,",");
		fprintf(header,"\n");
	}
	fprintf(header,"};\n\n");


	fprintf(header,"float out_w[N_OUT][N_HIDDEN + 1] = {\n");
	for(l = 0; l < N_OUT; l++){
		fprintf(header,"{ ");		
		for(d = 0; d < N_HIDDEN + 1; d++){
			if(INITIALIZE) {
				if(d == 0 || l == 52 || l == 53) fprintf(header,"%e ", 0.0);
				else fprintf(header,"%e ",WEIGHT * (rand()%100000));
			}
			else fprintf(header,"%e ",out_w[l][d]);
			if(d != N_HIDDEN) fprintf(header,",");
		}
		fprintf(header,"}");
		if(l != N_OUT - 1) fprintf(header,",\n");
	}
	fprintf(header,"};\n\n");
	return 0;
}

int pro_index(char *name){
	int i;
	for(i = 0; i < sizeof(phones)/sizeof(hmmType); i++){
		if(!strcmp(name,phones[i].name)){
			return i;
		}
	}
}
