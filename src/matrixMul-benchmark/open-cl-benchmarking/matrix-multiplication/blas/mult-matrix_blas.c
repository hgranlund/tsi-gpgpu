#include < sys/types.h >
#include < stdio.h >
#include < string.h >
#include < assert.h >
#include < clAmdBlas.h >
#include < math.h >
#define EPS 1.0e-15

//static const clAmdBlasOrder order = clAmdBlasColumnMajor;
static const clAmdBlasOrder order = clAmdBlasRowMajor;
static const clAmdBlasTranspose transA = clAmdBlasNoTrans;
static const clAmdBlasTranspose transB = clAmdBlasNoTrans;


extern "C" int  getPlatform(cl_platform_id *selected_platform);
extern "C" void checkStatus(const char *name,cl_int err);
extern "C" int  getDeviceInfo(cl_device_id device);
extern "C" void  checkResult(double *InMatA, double *InMatB, double *outMatC, int m,int n , int k);
extern "C" void setExeEnv ( cl_context *context, cl_uint *num_devices, cl_device_id **devices, cl_program *program);
extern "C" int   runAmdOpenCLDgemm(char *argv);
extern "C" void fill_dp_matrix(double* matrix,int rowSize,int colSize);


/****************************************************************
Function to return the execution time in seconds
****************************************************************/
double executionTime(cl_event &event)
{
    cl_ulong start, end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}


/****************************************************************
main() function
****************************************************************/
int main(int argc, char **argv)
{

        if (argc != 2 ){
                printf("\n Invalid number of arguments : < ./executable >  < matrix-size > \n");
                exit(-1);
        }
  if ( runAmdOpenCLDgemm(argv[1]))
    printf( "\n ...........Aborting \n");

  return 0;
}
int runAmdOpenCLDgemm(char *argv) {


        cl_uint                 num_devices; /*hold the number of devices */
        cl_device_id            *devices; /* hold list of devices */
        cl_context              context; /* hold the context */
        cl_platform_id          platform_id; /* Hold the OpenCL Platform Id */
        cl_int                  err; /* hold the err sode */
        cl_command_queue        cmd_queue;
  cl_mem      bufA, bufB, bufC;
  cl_event    event;

  int             rowA, rowB, rowC, colA, colB, colC; /* holds matrices dimensions */
        int             M, K, N, i, lda, ldb, ldc; /* holds matrices dimension and leading dimension */
        double          alpha = 1.0, beta = 0.0;
        double*         hMatA;  /* host matrix A */
        double*         hMatB;  /* host matrix B */
        double*         hMatC;  /* host matrix C */
        double*         dMatA;  /* device matrix A */
        double*         dMatB;  /* device matrix B */
        double*         dMatC;  /* device output matrix C */
  double    gflops=0.0;
  int   matrixSize;


         /* read matrices size from command line arguments */
   matrixSize = atoi(argv);
         M =  K = N = matrixSize ;


  rowA = M ; /* set rows of matrix A */
        colA = rowB = K ; /* set column of matrix A and rows of matrix B */
        colB = N ; /* set column of matrix B*/

        rowC = M ; /* set rows of resultant matrix C */
        colC = N ; /* set columns of resultant matrix C */

        lda = M ; ldb = K ; ldc = M; /* set the leading dimension of matrices */


  /**  Setting the exeution environment for openCL kernel **/
   setExeEnv ( &context , &num_devices, &devices,NULL);

         /******** 5. Create the command queue for requested devices *********/
         cmd_queue = clCreateCommandQueue( context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
         if( err != CL_SUCCESS || cmd_queue == 0)
         {
               printf("\n\t Failed to create command queue  \n" );
               exit (-1);
           }

  /* Setup clAmdBlas. */
      err = clAmdBlasSetup();
      if (err != CL_SUCCESS) {
          printf("clAmdBlasSetup() failed with %d\n", err);
          clReleaseCommandQueue(cmd_queue);
          clReleaseContext(context);
          return 1;
      }

        assert(hMatA = (double*) malloc (rowA * colA * sizeof(double)));
        assert(hMatB = (double*) malloc (rowB * colB * sizeof(double)));
        assert(hMatC = (double*) malloc (rowC * colC * sizeof(double)));

  /* Filling the Matrix with Values*/
  fill_dp_matrix(hMatA, rowA, colA);
  fill_dp_matrix(hMatB, rowB, colB);
        for (i=0; i < rowC * colC; i++){
                hMatC[i] = 0.0 ;
        }

   /* Prepare OpenCL memory objects and place matrices inside them. */
      bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, M * K * sizeof(*hMatA),
                          NULL, &err);
      bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, K * N * sizeof(*hMatB),
                          NULL, &err);
      bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, M * N * sizeof(*hMatC),
                          NULL, &err);

    err = clEnqueueWriteBuffer(cmd_queue, bufA, CL_TRUE, 0,
        M * K * sizeof(*hMatA), hMatA, 0, NULL, NULL);
      err = clEnqueueWriteBuffer(cmd_queue, bufB, CL_TRUE, 0,
        K * N * sizeof(*hMatB), hMatB, 0, NULL, NULL);
      err = clEnqueueWriteBuffer(cmd_queue, bufC, CL_TRUE, 0,
        M * N * sizeof(*hMatC), hMatC, 0, NULL, NULL);

   /* Call clAmdBlas function. */
      err = clAmdBlasDgemm(order, transA, transB, M, N, K, alpha, bufA,
                         lda, bufB, ldb, beta, bufC, ldc, 1, &cmd_queue,
                         0, NULL, &event);
      if (err != CL_SUCCESS) {
          printf("clAmdBlasSgemm() failed with %d\n", err);
          return 1;
      }
      else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(cmd_queue, bufC, CL_TRUE, 0, M * N * sizeof(*hMatC),
                                  hMatC, 0, NULL, NULL);

  gflops = 1.0e-9 * ((2.0 * M * N * K) / executionTime(event));

        printf("\n Matrix Size \t  Execution time \t  Gflops ");
  printf("\n -------------------------------------------------------------------------\n");
        printf("\n (%d * %d )  \t  %.5f sec   \t    %.5f \n", M,K, executionTime(event), gflops);
        /* At this point you will get the result of SGEMM placed in C array. */
      }

  /* check CPU+GPU results against CPU results */
        checkResult( hMatA, hMatB, hMatC,M,N,K);

      /* Release OpenCL memory objects. */
      clReleaseMemObject(bufC);
      clReleaseMemObject(bufB);
      clReleaseMemObject(bufA);

      /* Finalize work with clAmdBlas. */
      clAmdBlasTeardown();

      /* Release OpenCL working objects. */
      clReleaseCommandQueue(cmd_queue);
      clReleaseContext(context);

    return 0;
}

void setExeEnv ( cl_context *context, cl_uint *num_devices, cl_device_id **devices, cl_program *program) {

  cl_platform_id  platform_id; /* Hold the OpenCL Platform Id */
  cl_int    err; /* hold the err sode */
  int   count; /* loop control variable */

  /** 2. Get the OpenCL Platform ID ****/
        if ( getPlatform(&platform_id)) {
                printf(" \n\t\t Failed to get platform Info \n");
                exit(-1);
        }

         /** 3. Get the count & list of available OpenCL devices  */
        err = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 0, 0, num_devices);
        if( err != CL_SUCCESS  || *num_devices == 0) {
                 printf("\n\t\t ERROR : Failed to get Device Ids Or No OpenCL device found  \n");
                 exit(-1);
         }
         else {
                   assert( (*devices = (cl_device_id *) malloc( sizeof(cl_device_id ) * (*num_devices))) != NULL);
                   err = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, (*num_devices), *devices, 0);
                   checkStatus ("clGetDeviceIDs Failed", err);
         }

        if( getDeviceInfo ( (*devices)[0]) )
                        printf("\n\t Failed to print the device info \n");

         /****** 4. Create Context for GPU Devices ***********/
        *context = clCreateContext( NULL, 1, devices[0], 0, 0, &err);
         if ( err != CL_SUCCESS || context == 0)
        {
                printf("\n\t No GPU detected ");
                printf("\n\t Context : %d , Err : %d",context, err);
                exit(-1);
        }




}
void fill_dp_matrix(double* matrix,int rowSize,int colSize)
{

       int     row_count, col_count ;

        for( row_count=0; row_count < rowSize; row_count++)
             for( col_count=0; col_count < colSize; col_count++)
                        matrix[row_count * colSize + col_count] = drand48();

}



int getPlatform(cl_platform_id *selected_platform)
{
        cl_int          err;
        int             count;
        char            pbuff[100];
        cl_uint         num_platforms, num_devices;
        cl_platform_id  *platforms;

        *selected_platform = NULL;

 /*  Get the number of OpenCL Platforms Available */
        err = clGetPlatformIDs ( 0, 0, &num_platforms);
        if( err != CL_SUCCESS || num_platforms == 0) {
                printf(" \n\t\t No Platform Found \n");
                return 1;
        }
        else
        {
                if( num_platforms == 0)
                {
                     return 1;
                }
               else
                {
                        /* Allocate the space for available platform*/
                        assert( (platforms = (cl_platform_id *) malloc( sizeof(cl_platform_id) * (num_platforms))) != NULL);
                        /*  Get available OpenCL Platforms IDs*/
                        err = clGetPlatformIDs( num_platforms, platforms, NULL);
                        checkStatus(" Failed to get Platform IDs",err);
                        for ( count = 0 ; count < num_platforms ; count++)
                        {
                                /* get platform info*/
                                err=clGetPlatformInfo(platforms[count],CL_PLATFORM_NAME,sizeof(pbuff),pbuff,NULL);
                                checkStatus ("clGetPlatformInfo Failed ",err);
                                /* get device id and info*/
                                err = clGetDeviceIDs( platforms[count], CL_DEVICE_TYPE_GPU,0,0,&num_devices);
                                if( err != CL_SUCCESS  || num_devices == 0)
                                {
                                         continue;
                                }
                                else
                                {
                                        /* get selected platform*/
                                        *selected_platform = platforms[count];
                                        printf("\n Platform used    :  %s",pbuff);
                                        break;
                                }
                        }
                }
        }

        if ( count == num_platforms ) {
                printf(" \n\t No platform found \n");
                return 1;
        }

 free(platforms);
        return 0;
}
int getDeviceInfo( cl_device_id device)
{

        int                     icount;
        char                    dbuff[100];
        cl_uint                 device_add_bits;
        cl_uint                 device_max_comp_unit;   /* Device maximum compute units */
        cl_uint                  device_max_freq;        /* Device maximum clock frequency */
        cl_device_type           type;
        cl_int                   err;
        cl_ulong                 device_global_mem;        /* Hold Device memory size */
        cl_ulong                 device_cache_size;        /* Hold Device memory size */
        cl_ulong                 device_local_mem;        /* Hold Device memory size */
        cl_device_mem_cache_type device_cache_type; /* Hold the device cache type like read / write etc*/


       /* Get device Name */
       err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dbuff), &dbuff, NULL);
       checkStatus("clGetDeviceInfo Failed ",err);
       printf("\n OpenCL Device    : %s  ",dbuff);

       /* Get device type  */
       err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
       checkStatus("clGetDeviceInfo Failed ",err);
       switch ( type )
       {
              case CL_DEVICE_TYPE_CPU : printf("\n Device Type\t\t\t : CPU ");
              break;
              case CL_DEVICE_TYPE_GPU : printf("\n Device Type      : GPU");
              break;
              case CL_DEVICE_TYPE_ACCELERATOR:printf("\n Device Type\t\t\t : Dedicated OpenCLAccelerator");
              break;
              case CL_DEVICE_TYPE_DEFAULT : printf("\n Device Type\t\t\t: DEFAULT  ");
              break;
        }

        /* Get device  global memory in bytes */
         err = clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_global_mem),
                                &device_global_mem, NULL);
          checkStatus("clGetDeviceInfo Failed ",err);
          printf("\n Global Memory    : %lf MB  ",(double)(device_global_mem)/(1024 * 1024));

          /* Get device cache type */
          err = clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                                 sizeof(device_cache_type),
                                  &device_cache_type, NULL);
          checkStatus("clGetDeviceInfo Failed ",err);

          switch ( device_cache_type )
          {
                case CL_NONE : printf(" \( Cache Type : None ");
                break;
                case CL_READ_ONLY_CACHE : printf(" \( Cache Type: ro  ");
                break;
                case CL_READ_WRITE_CACHE : printf(" \( Cache Type : rw  ");
                break;
          }
 /* Get device cache size */
         err = clGetDeviceInfo(  device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                                   sizeof(device_cache_size),
                                   &device_cache_size, NULL);
         checkStatus("clGetDeviceInfo Failed ",err);
         printf(" , Cache Size     : %ld Bytes ) ",device_cache_size);

          /* Get device local memory size in bytes */
          err = clGetDeviceInfo( device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_local_mem),
                                 &device_local_mem, NULL);
          checkStatus("clGetDeviceInfo Failed ",err);
          printf(" \n Device Local Memory:%lf KB  \n\n",(double)device_local_mem/(1024));

          return 0 ;

}
inline void checkStatus(const char *name,cl_int err)
{
        if (err != CL_SUCCESS)
        {

                printf("\n\t\t Error: %s (%d) \n",name, err);
                exit(-1);
        }
}
void checkResult(double *InMatA, double *InMatB, double *outMatC, int m, int n , int k )
{
        double  alpha = 1.0, beta = 0.0;
        int     i;
        int     j;
        int     k1;
        double  *tempOut;
        double  errorNorm = 0.0;
        double  eps=EPS;
        double  relativeError=0.0;
        int     flag=0;

        tempOut  = (double*) malloc (m * n  * sizeof(double));
        if (tempOut == 0){
        printf("\n Memory allocation Failed for Resultant Matrix");
        exit (0);
        }

        /* CPU Compuation Performs operation using CBLAS */
       // cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, m, n , k, alpha, InMatA, m , InMatB , k, beta, tempOut, m);

        /******************************************************************
        Serial computation
        uncomment the below section if want to do the CPU computation
        using i,j,k loop method. Method work only for square matrices.
         *******************************************************************/
  if ( order == clAmdBlasColumnMajor ) {
          for (i = 0; i < n  ; ++i) {
                  for (j = 0; j < n; ++j) {
                        double  cprod = 0;
                        for (k1 = 0; k1 < n; ++k1) {
                                cprod += InMatA[k1 * n + i] * InMatB[j * n + k1];
                        }
                  tempOut[j * n + i] = alpha * cprod + beta * tempOut[j * n + i];

                  }
          }
  }

  if ( order == clAmdBlasRowMajor ) {
     for (i = 0; i < n  ; ++i) {
                  for (j = 0; j < n; ++j) {
                        double  cprod = 0;
                        for (k1 = 0; k1 < n; ++k1) {
                                cprod += InMatA[i * n + k1] * InMatB[k1 * n + j];
                        }
                  tempOut[i * n + j] = alpha * cprod + beta * tempOut[i * n + j];
                  }
          }
  }



        /*** check relative error with approx precision ****/
        for( i = 0; i < m*n; ++i) {

                if (fabs(tempOut[i]) > fabs(outMatC[i]))
                        relativeError = fabs((tempOut[i] - outMatC[i]) / tempOut[i]);
                else
                        relativeError = fabs((outMatC[i] - tempOut[i]) / outMatC[i]);

                if (relativeError > eps && relativeError != 0.0e+00 ){
                        if(errorNorm < relativeError) {
                        errorNorm = relativeError;
                        flag=1;
                        }
                }

        }
        if( flag == 1) {
  printf(" \n Results verfication : Failed");
                printf(" \n Considered machine precision : %e", eps);
                printf(" \n Relative Error                  : %e \n", errorNorm);

        }
        else {

                printf("\n Results verfication : Success \n");
        }

        free(tempOut);
}
