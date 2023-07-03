/* This file should be almost identical to ex2.cu from homework 2. */
/* once the TODOs in this file are complete, the RPC version of the server/client should work correctly. */

#include "ex3.h"
#include <cuda/atomic>
#include "ex2.h"


using cuda::memory_order_relaxed;
using cuda::memory_order_acquire;
using cuda::memory_order_release;


__device__ void prefixSum(int arr[], int len, int tid, int threads) {
    // TODO complete according to hw1
    int increment;

    for (int stride = 1; stride < len; stride *= 2) {
        if (tid >= stride && tid < len) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < len) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

__device__ void argmin(int arr[], int len, int tid, int threads) {
    assert(threads == len / 2);
    int halfLen = len / 2;
    bool firstIteration = true;
    int prevHalfLength = 0;
    while (halfLen > 0) {
        if(tid < halfLen){
            if(arr[tid] == arr[tid + halfLen]){ //a corenr case
                int lhsIdx = tid;
                int rhdIdx = tid + halfLen;
                int lhsOriginalIdx = firstIteration ? lhsIdx : arr[prevHalfLength + lhsIdx];
                int rhsOriginalIdx = firstIteration ? rhdIdx : arr[prevHalfLength + rhdIdx];
                arr[tid + halfLen] = lhsOriginalIdx < rhsOriginalIdx ? lhsOriginalIdx : rhsOriginalIdx;
            }
            else{ //the common case
                bool isLhsSmaller = (arr[tid] < arr[tid + halfLen]);
                int idxOfSmaller = isLhsSmaller * tid + (!isLhsSmaller) * (tid + halfLen);
                int smallerValue = arr[idxOfSmaller];
                int origIdxOfSmaller = firstIteration * idxOfSmaller + (!firstIteration) * arr[prevHalfLength + idxOfSmaller];
                arr[tid] = smallerValue;
                arr[tid + halfLen] = origIdxOfSmaller;
            }
        }
        __syncthreads();
        firstIteration = false;
        prevHalfLength = halfLen;
        halfLen /= 2;
    }
}

__device__ void colorHist(uchar img[][CHANNELS], int histograms[][LEVELS]){
    int tid = threadIdx.x;;
    int threads = blockDim.x;

    //fill histogram arrays with zeros
    int entriesCount = CHANNELS * LEVELS;
    for (int i = tid; i < entriesCount; i += threads){
        ((int*)histograms)[i] = 0;
    }

    __syncthreads();

    //three histograms - one for each channel
    int* greenHist = histograms[0];
    int* blueHist = histograms[1];
    int* redHist = histograms[2];

    //increment histograms according to value of each pixel
    for(int i = tid; i < SIZE * SIZE; i += threads){
        uchar g, b, r;
        g = img[i][0];
        b = img[i][1];
        r = img[i][2];
        atomicAdd_block(&greenHist[g], 1);
        atomicAdd_block(&blueHist[b], 1);
        atomicAdd_block(&redHist[r], 1);
    }
}

__device__ void performMapping(uchar maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS]){
    int tid = threadIdx.x;
    int threads = blockDim.x;

    for(int i = tid; i < SIZE * SIZE; i += threads){
        uchar g, b, r;
        g = targetImg[i][0];
        b = targetImg[i][1];
        r = targetImg[i][2];
        uchar newG = maps[0][g];
        uchar newB = maps[1][b];
        uchar newR = maps[2][r];
        resultImg[i][0] = newG;
        resultImg[i][1] = newB;
        resultImg[i][2] = newR;
    }
}

__device__
void process_image(uchar *target, uchar *reference, uchar *result) {
    //TODO complete according to HW2
    //(This file should be almost identical to ex2.cu from homework 2.)

    //maximum total shared memory: 15KB
    __shared__ int targetHistogram[CHANNELS][LEVELS];
    __shared__ int referenceHistogram[CHANNELS][LEVELS];
    __shared__ uchar maps[CHANNELS][LEVELS];
    extern __shared__ int arrs[]; //arrs[threadCount / (LEVELS / 2)][LEVELS]    ,max size: 8KB
    

    int tid = threadIdx.x;
    int threads = blockDim.x;

    //get color histograms
    colorHist((uchar (*)[CHANNELS])target, targetHistogram);
    colorHist((uchar (*)[CHANNELS])reference, referenceHistogram);
    
    __syncthreads();

    int threadsForScan = LEVELS;
    int concurentScanCout = max(1, threads / threadsForScan);
    int scanCount = 2* CHANNELS; //2 images
    int threadPadding = (scanCount % concurentScanCout) * threadsForScan;

    for(int firstScanId = 0; firstScanId < scanCount; firstScanId += concurentScanCout){
        //preform concurentScanCout scan operations concurently on histograms
        int concurentScanId = tid / threadsForScan;
        int scanId = firstScanId + concurentScanId;
        int col = tid % threadsForScan;
        bool firstImg = scanId < CHANNELS;
        int channel = scanId % CHANNELS;
        int *hist = firstImg ? targetHistogram[channel] : referenceHistogram[channel];
        bool scanWithPadding = scanId >= (scanCount - 1);
        int paddingShift = (scanId >= scanCount) * (scanId - scanCount + 1) * threadsForScan;
        int num_threads = min(threads, threadsForScan) + scanWithPadding * threadPadding;   
        prefixSum(hist, LEVELS, col + paddingShift, num_threads);
    }

    __syncthreads();

    int threadsForReduce = LEVELS / 2;
    int concurentReduceCout = threads / threadsForReduce;
    int reduceCount = CHANNELS * LEVELS;

    for(int firstRow = 0; firstRow < reduceCount; firstRow += concurentReduceCout){

        for (int i = tid; i < concurentReduceCout * LEVELS; i+= threads){
            int arr = i / LEVELS;
            int row = firstRow + arr;
            int channel = row / LEVELS;
            int i_target = row % LEVELS;
            int i_reference = i % LEVELS;
            (arrs + arr * LEVELS)[i_reference] = abs(referenceHistogram[channel][i_reference] - targetHistogram[channel][i_target]);
        }

        __syncthreads();

        //preform concurentReduceCout reduce operations concurently on arrs
        int arr = tid / threadsForReduce ;
        int row = firstRow + arr;
        if (row < reduceCount){
            int col = tid % threadsForReduce;
            argmin((arrs + arr * LEVELS), LEVELS, col, threadsForReduce);
            if(col == 0){
                int channel = row / LEVELS;
                int mapInput = row % LEVELS;
                int argMin = (arrs + arr * LEVELS)[1];
                maps[channel][mapInput] = argMin;
            }
        }
        __syncthreads();
    }

    performMapping(maps, (uchar(*)[CHANNELS])target, (uchar(*)[CHANNELS])result);
}


__global__
void process_image_kernel(uchar *target, uchar *reference, uchar *result){
    process_image(target, reference, result);
}


// TODO complete according to HW2:
//          implement a SPSC queue,
//          implement the persistent kernel,
//          implement a function for calculating the threadblocks count
// (This file should be almost identical to ex2.cu from homework 2.)



// Code assumes it is a power of two
#define NSLOTS 16


struct Entry {
    int job_id;
    uchar* target;
    uchar* reference;
    uchar* img_out;
    uchar* remote_img_out;
};

struct queue
{
    Entry data[NSLOTS];
    char pad_a[128];
    cuda::atomic<int> pi;
    char pad_b[128];
    cuda::atomic<int> ci;
    char pad_c[128];
    cuda::atomic<bool> kill;

    queue():
    pi(0), ci(0), kill(false) {}

    __host__ __device__ bool pop(Entry *ent)
    {
        int cur_pi = pi.load(memory_order_acquire);
        int cur_ci = ci.load(memory_order_relaxed);
        if ((cur_pi-cur_ci) == 0){
            return false;
        }
        *ent = data[cur_ci & (NSLOTS - 1)];
        ci.store(cur_ci+1, memory_order_release);
        
        return true;
    }

    __host__ __device__ bool push(Entry *ent)
    {
        int cur_ci = ci.load(memory_order_acquire);
        int cur_pi = pi.load(memory_order_relaxed);
        if ((cur_pi-cur_ci) == NSLOTS){
            return false;
        }
        data[cur_pi & (NSLOTS - 1)] = *ent;
        pi.store(cur_pi+1, memory_order_release);        
        return true;
    }
};

__global__ void gpu_process_image_consumer(queue *cpu_to_gpu_qeueus, queue *gpu_to_cpu_qeueus) {
    queue *h2g = &(cpu_to_gpu_qeueus[blockIdx.x]);
    queue *g2h = &(gpu_to_cpu_qeueus[blockIdx.x]);

    int tid = threadIdx.x;
    __shared__ Entry entry;
    __shared__ bool kill;

    if (tid == 0)
        kill = false;
        
    __syncthreads();

    while(true){
        if (tid == 0){
            while(!h2g->pop(&entry)){
                bool cur_kill = h2g->kill.load(memory_order_relaxed);
                if (cur_kill){
                    kill = true;
                    break;
                }
            }
        }
        __syncthreads();
        if(kill){
            break;
        }
        process_image(entry.target, entry.reference, entry.img_out);
        __syncthreads();
        if(tid ==0){
            Entry out_entry = {
                .job_id = entry.job_id,
                .img_out = entry.img_out,
                .remote_img_out = entry.remote_img_out};
            while(!g2h->push(&out_entry)){}
        }

    }
}


class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)


public:
    //TODO complete according to HW2
    //(This file should be almost identical to ex2.cu from homework 2.)

    int blocks;
    queue *cpu_to_gpu_queues;
    queue *gpu_to_cpu_queues;
    char *queue_buffer;
    int next_block = 0;
    int threadsPerBlock, arrsSizeBytes;

    queue_server(int threads) {
        blocks = calc_blocks(threads);
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)

        // TODO initialize host state
        CUDA_CHECK(cudaMallocHost(&queue_buffer, blocks * (sizeof(queue) + sizeof(queue))));

        cpu_to_gpu_queues = new (queue_buffer) queue[blocks];
        gpu_to_cpu_queues = new (queue_buffer + sizeof(queue[blocks])) queue[blocks];

        int concurentReduceCount = threads / (LEVELS / 2);
        arrsSizeBytes = sizeof(int) * concurentReduceCount * LEVELS;

        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
        gpu_process_image_consumer<<<blocks, threads, arrsSizeBytes>>>(cpu_to_gpu_queues, gpu_to_cpu_queues);
    }

    ~queue_server() override
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)

        for(int i=0; i<blocks; i++){
            cpu_to_gpu_queues[i].kill.store(true, memory_order_relaxed);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cpu_to_gpu_queues->~queue();
        gpu_to_cpu_queues->~queue();
        CUDA_CHECK(cudaFreeHost(queue_buffer));
    }

    bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
        auto &next = cpu_to_gpu_queues[job_id % blocks];
        Entry new_entry;
        new_entry.job_id = job_id;
        new_entry.target = target;
        new_entry.reference = reference;
        new_entry.img_out = result;
        return next.push(&new_entry);
    }

    bool dequeue(int *job_id) override
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
        int block = next_block;
        for (int i = 0; i < blocks; ++i, ++block) {
            if (block == blocks){
                block = 0;
            }
            Entry entry;
            if(gpu_to_cpu_queues[block].pop(&entry)){
                // TODO return the job_id of the request that was completed.
                *job_id = entry.job_id;
                next_block = block - 1;
                if (next_block < 0){
                    next_block += blocks;
                }
                return true;
            }
        }
        return false;
    }

    
    int calc_blocks(int threads_per_block)
    {
        int device;
        cudaDeviceProp prop;

        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        int maxByRegsPerSM = prop.regsPerMultiprocessor / threads_per_block / 32;
        int maxBySharedMemory = prop.sharedMemPerMultiprocessor / (6912 + 8 * threads_per_block);
        int maxByThreads = prop.maxThreadsPerMultiProcessor / threads_per_block;

        printf("maxByRegsPerSM: %d\nmaxBySharedMemoryPerSM: %d\nmaxByThreadsPerSM: %d\n", maxByRegsPerSM, maxBySharedMemory, maxByThreads);

        auto blocks = min(min(maxByRegsPerSM, maxBySharedMemory), maxByThreads) * prop.multiProcessorCount;
        printf("number of blocks: %d\n", blocks);
        return blocks;
    }
};


std::unique_ptr<queue_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
