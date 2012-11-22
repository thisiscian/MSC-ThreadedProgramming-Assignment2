#include <stdio.h>
#include <math.h>


#define N 729
//#define reps 100 
#define reps 100
#include <omp.h> 

double a[N][N], b[N][N], c[N];
int jmax[N];  


void init1(void);
void init2(void);
void runloop(int); 
void loop1chunk(int, int);
void loop2chunk(int, int);
void valid1(void);
void valid2(void);


int main(int argc, char *argv[]) { 

  double start1,start2,end1,end2;
  int r;

  init1(); 

  start1 = omp_get_wtime(); 

  for (r=0; r<reps; r++){ 
    runloop(1);
  } 

  end1  = omp_get_wtime();  

  valid1(); 

  printf("Total time for %d reps of loop 1 = %f\n",reps, (float)(end1-start1)); 


  init2(); 

  start2 = omp_get_wtime(); 

  for (r=0; r<reps; r++){ 
    runloop(2);
  } 

  end2  = omp_get_wtime(); 

  valid2(); 

  printf("Total time for %d reps of loop 2 = %f\n",reps, (float)(end2-start2)); 

} 

void init1(void){
  int i,j; 

  for (i=0; i<N; i++){ 
    for (j=0; j<N; j++){ 
      a[i][j] = 0.0; 
      b[i][j] = 3.142*(i+j); 
    }
  }

}

void init2(void){ 
  int i,j, expr; 

  for (i=0; i<N; i++){ 
    expr =  i%( 3*(i/30) + 1); 
    if ( expr == 0) { 
      jmax[i] = N;
    }
    else {
      jmax[i] = 1; 
    }
    c[i] = 0.0;
  }

  for (i=0; i<N; i++){ 
    for (j=0; j<N; j++){ 
      b[i][j] = (double) (i*j+1) / (double) (N*N); 
    }
  }
 
} 


void runloop(int loopid)  {
	int global_work_remaining[omp_get_num_threads()];
#pragma omp parallel default(none) shared(global_work_remaining, loopid) 
  {
    int my_id  = omp_get_thread_num();
    int nthreads = omp_get_num_threads(); 
    int ipt = (int) ceil((double)N/(double)nthreads); 
		int segment_id = my_id;
    int segment_lo = segment_id*ipt;
    int segment_hi = (segment_id+1)*ipt;
    if (segment_hi > N) segment_hi = N;
		int segment_range = segment_hi-segment_lo;
//		printf("segment %d: from %d to %d\n", segment_id, segment_lo, segment_hi);
		int i;
		#pragma omp critical
		{
			global_work_remaining[my_id] = segment_range;
		}
		int local_lo, local_hi, local_work;
		int finished = 0;
		while(!finished)
		{
			#pragma omp critical
			{
				int old_id = segment_id;
				if(global_work_remaining[segment_id] == 0)
				{
					for(i=0; i<nthreads; i++)
					{
						if(global_work_remaining[segment_id] < global_work_remaining[i])
						{
							segment_id = i;
						}
					}
					if(old_id == segment_id || nthreads == 1)
					{
						finished = 1;
					}
					else
					{
					//	printf("%d: switched from segment %d to %d\n", my_id, old_id, segment_id);
				    segment_lo = segment_id*ipt;
				    segment_hi = (segment_id+1)*ipt;
  				  if (segment_hi > N) segment_hi = N;
						segment_range = global_work_remaining[segment_id];
					}
				}
				else
				{
					segment_range = global_work_remaining[segment_id];
				}
				local_work = segment_range/nthreads;
				if(local_work < 1 && finished == 0) local_work = 1;
				global_work_remaining[segment_id] -= local_work;
//				printf("work left%d = from %d - %d\n", segment_id, segment_hi-segment_range+local_work, segment_hi);
				
			}
			if(finished == 0)
			{
				local_lo = segment_hi - segment_range;
				local_hi = local_lo +	local_work;
				//printf("%d to %d\n", local_lo, local_hi);
	    	switch (loopid) { 
	    	   case 1: loop1chunk(local_lo,local_hi); break;
	    	   case 2: loop2chunk(local_lo,local_hi); break;
	    	} 
			}
		}
  }
}

void loop1chunk(int lo, int hi) { 
  int i,j; 
  
  for (i=lo; i<hi; i++){ 
    for (j=N-1; j>i; j--){
      a[i][j] += cos(b[i][j]);
    } 
  }

} 



void loop2chunk(int lo, int hi) {
  int i,j,k; 
  double rN2; 

  rN2 = 1.0 / (double) (N*N);  

  for (i=lo; i<hi; i++){ 
    for (j=0; j < jmax[i]; j++){
      for (k=0; k<j; k++){ 
	c[i] += (k+1) * log (b[i][j]) * rN2;
      } 
    }
  }

}

void valid1(void) { 
  int i,j; 
  double suma; 
  
  suma= 0.0; 
  for (i=0; i<N; i++){ 
    for (j=0; j<N; j++){ 
      suma += a[i][j];
    }
  }
  printf("Loop 1 check: Sum of a is %lf\n", suma);

} 


void valid2(void) { 
  int i; 
  double sumc; 
  
  sumc= 0.0; 
  for (i=0; i<N; i++){ 
    sumc += c[i];
  }
  printf("Loop 2 check: Sum of c is %f\n", sumc);
} 
 

