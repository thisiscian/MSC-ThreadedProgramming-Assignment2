#include <stdio.h>
#include <math.h>


#define N 729
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
	int global_work_remaining[omp_get_max_threads()];
	#pragma omp parallel default(none) shared(global_work_remaining, loopid, a, b, c) 
  {
		int i;

    int my_id  = omp_get_thread_num();
    int nthreads = omp_get_num_threads(); 
    int ipt = (int) ceil((double)N/(double)nthreads); 
		
		/* there should be as many chunks as there are threads
		 * and they should have roughly identical ranges		*/			
		int chunk_id = my_id;
    int chunk_lo = chunk_id*ipt;
   	int chunk_hi = (chunk_id+1)*ipt;
    if (chunk_hi > N) chunk_hi = N;
		int chunk_range = chunk_hi-chunk_lo;

		/* these are the variables that tell how much
		 * work a thread is doing in a chunk */
		int local_lo, local_hi, local_work;

		/* initialise the shared array*/
		global_work_remaining[my_id] = chunk_range;
		#pragma omp barrier

		/* because you can't immediately break in a critical section
		 * a flag is needed to indicate to break immediately after */
		int finished = 0;

		/* continue to do work unless there is no work left to do */
		while(1)
		{

			#pragma omp critical
			{
				/* if the thread has no work left, look for more work */
				if(global_work_remaining[chunk_id] == 0)
				{
					int old_id = chunk_id;
					/* loop over all array elements to see who is most loaded */
					for(i=0; i<nthreads; i++)
					{
						if(global_work_remaining[chunk_id] < global_work_remaining[i])
						{
							chunk_id = i;
						}
					}
					/* if no job has more work to do than the current thread,
					 * and the current thread has no work, then there is no work
					 * left, so break out of the while loop and finish the function */
					if(old_id == chunk_id)
					{
						finished = 1; // raise flag to indicate to break
						chunk_range = 0; //ensure shared array is unchanged
					}
					/* otherwise, update your chunk values, and do some work */
					else
					{
				    chunk_hi = (chunk_id+1)*ipt;
  				  if (chunk_hi > N) chunk_hi = N;
						chunk_range = global_work_remaining[chunk_id];
					}
				}
				else
				{
					chunk_range = global_work_remaining[chunk_id];
				}
				/* local work to do is a fraction of the chunk size
 				 * thread should do at least 1 iteration, however	*/
				local_work = floor((double)chunk_range/(double)nthreads);
				if(local_work < 1 && finished == 0) local_work = 1;
				global_work_remaining[chunk_id] -= local_work;
			}
			/* all interactions with the shared array are over
			 * if there is no more work, leave the while loop,
			 * otherwise start working on the loop */
			if(finished == 1) break;
			local_lo = chunk_hi - chunk_range;
			local_hi = local_lo +	local_work;
	   	switch (loopid) { 
	   	   case 1: loop1chunk(local_lo,local_hi); break;
	   	   case 2: loop2chunk(local_lo,local_hi); break;
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
 

