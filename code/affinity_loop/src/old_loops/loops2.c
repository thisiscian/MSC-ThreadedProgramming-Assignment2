#include <stdio.h>
#include <math.h>

#define N 729
#define reps 100
//#define reps 100 
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
	setbuf(stdout, NULL);	
  init1(); 

  start1 = omp_get_wtime(); 

  for (r=0; r<reps; r++){ 
    runloop(1);
		printf("iteration %d went fine\n", r);
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


void runloop(int loopid)
{
	int local_iterations_remaining[omp_get_num_threads()];
	#pragma omp parallel default(none) shared(loopid, local_iterations_remaining) 
	{
		int i, my_lo, my_hi, iterations_to_do;
		omp_lock_t writelock;
		omp_init_lock(&writelock);
		int myid  = omp_get_thread_num();
  	int nthreads = omp_get_num_threads(); 
  	  
		int ipt = (int) ceil((double)N/(double)nthreads);
		int current_work_segment = myid;
		int lo, hi;
		#pragma omp single
		{
			for(i=0; i<nthreads; i++)
			{
				lo = i*ipt;
 				hi = (i+1)*ipt;
  			if (hi > N) hi = N;
				local_iterations_remaining[i] = hi-lo;	// number of iterations remaining
			}
		}
 		hi = (current_work_segment+1)*ipt;
  	if (hi > N) hi = N;
		while(1)
		{
			int old_work_segment = current_work_segment;
			omp_set_lock(&writelock);
			printf("\t\tI AM WRITING NOW: %d\n", myid);
			if(local_iterations_remaining[current_work_segment] <= 0)
			{
				for(i=0; i<nthreads; i++)
				{
					if(local_iterations_remaining[i] > local_iterations_remaining[current_work_segment])
					{
						current_work_segment = i;
					}
				}
				if(old_work_segment == current_work_segment)
				{
					printf("\t%d is out of work (%d)\n", myid, local_iterations_remaining[current_work_segment]);
					omp_unset_lock(&writelock);
					break;
				}
				printf("\t%d is now working in segment %d\n", myid, current_work_segment);
				hi = (current_work_segment+1)*ipt;
				if(hi > N) hi = N;
				omp_unset_lock(&writelock);
				printf("\t\tI AM STOPPED NOW: %d\n", myid);
			}
			else if(local_iterations_remaining[current_work_segment] > 0)
			{
				iterations_to_do = local_iterations_remaining[current_work_segment]/nthreads;
				if(iterations_to_do < 1) iterations_to_do = 1;
				my_lo = hi-local_iterations_remaining[current_work_segment];
				my_hi = iterations_to_do+my_lo;
				#pragma omp atomic
					local_iterations_remaining[current_work_segment] -= iterations_to_do;
				if(local_iterations_remaining[current_work_segment] < 0)
				{
					printf("something is very amiss here: %d \n", local_iterations_remaining[current_work_segment]); 
				}
	
				omp_unset_lock(&writelock);
				printf("\t\tI AM STOPPED NOW: %d\n", myid);
		 		switch (loopid)
				{ 
	  		  case 1: loop1chunk(my_lo,my_hi); break;
	  		  case 2: loop2chunk(my_lo,my_hi); break;
	  		}
			}
		}
		printf("%d at barrier\n", myid);
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
