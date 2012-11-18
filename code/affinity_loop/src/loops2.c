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
		int local_iterations_remaining[omp_get_num_threads()];
#pragma omp parallel default(none) shared(loopid, local_iterations_remaining) 
  {
		omp_lock_t writelock;
		omp_init_lock(&writelock);
		//initialise the loop
    int myid  = omp_get_thread_num();
    int nthreads = omp_get_num_threads(); 
		
		int working_on_thread = myid; // variable which indicates which segment the thread is working on
    
		int ipt = (int) ceil((double)N/(double)nthreads);
		int current_work_segment = myid;
   	int lo = current_work_segment*ipt;
   	int hi = (current_work_segment+1)*ipt;
   	if (hi > N) hi = N;

		while(1)
		{
			if(omp_test_lock(&writelock))
			{
				local_iterations_remaining[current_work_segment] = hi-lo;	// number of iterations remaining
				omp_unset_lock(&writelock);
				break;
			}
		}	
		
		while(1)
		{
			hi = (current_work_segment+1)*ipt;
			if(hi > N) hi = N;
			while(1)
			{
				if(omp_test_lock(&writelock))
				{
					if(local_iterations_remaining[current_work_segment] <= 0)
					{
						omp_unset_lock(&writelock);
						break;
					}
					// split up the work per processor working
					int iterations_to_do = local_iterations_remaining[current_work_segment]/nthreads;
					if(iterations_to_do < 1) iterations_to_do = 1;
					int my_lo = hi-local_iterations_remaining[current_work_segment];
					int my_hi = iterations_to_do+my_lo;
					local_iterations_remaining[current_work_segment] -= iterations_to_do;
					if(local_iterations_remaining[current_work_segment] < 0) local_iterations_remaining[current_work_segment] = 0;
					omp_unset_lock(&writelock);
					
		  	 	switch (loopid)
					{ 
	   		    case 1: loop1chunk(my_lo,my_hi); break;
	   		    case 2: loop2chunk(my_lo,my_hi); break;
	   			} 
		 		}
			}
			int old_work_segment = current_work_segment;
			while(1)
			{
				if(omp_test_lock(&writelock))
				{
					for(int i=0; i<nthreads; i++)
					{
						if(local_iterations_remaining[i] > local_iterations_remaining[current_work_segment])
						{
							current_work_segment = i;
						}
					}
					omp_unset_lock(&writelock);
					break;
				}	
			}
			if(old_work_segment == current_work_segment)
			{
				break;
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
 

