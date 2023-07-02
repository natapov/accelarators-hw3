/* CUDA 10.2 has a bug that prevents including <cuda/atomic> from two separate
 * object files. As a workaround, we include ex2.cu directly here. */
#include "ex2.cu"

#include <cassert>
#include <vector>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <infiniband/verbs.h>
#define NSLOTS 16
auto my_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
class server_rpc_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> gpu_context;
    std::array<int, OUTSTANDING_REQUESTS> input_read_done_count;


public:
    explicit server_rpc_context(uint16_t tcp_port) : rdma_server_context(tcp_port),
        gpu_context(create_queues_server(256))
    {
        std::fill(input_read_done_count.begin(), input_read_done_count.end(), 0);

    }

    virtual void event_loop() override
    {
        /* so the protocol goes like this:
         * 1. we'll wait for a CQE indicating that we got an Send request from the client.
         *    this tells us we have new work to do. The wr_id we used in post_recv tells us
         *    where the request is.
         * 2. now we send an RDMA Read to the client to retrieve the request.
         *    we will get a completion indicating the read has completed.
         * 3. we process the request on the GPU.
         * 4. upon completion, we send an RDMA Write with immediate to the client with
         *    the results.
         */
        rpc_request* req;
        uchar *img_target, *img_reference;
        uchar *img_out;

        bool terminate = false, got_last_cqe = false;

        while (!terminate || !got_last_cqe) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		        VERBS_WC_CHECK(wc);

                switch (wc.opcode) {
                case IBV_WC_RECV:
                    /* Received a new request from the client */
                    req = &requests[wc.wr_id];
                    img_target = &images_target[wc.wr_id * IMG_BYTES];
                    img_reference = &images_reference[wc.wr_id * IMG_BYTES];
                    /* Terminate signal */
                    if (req->request_id == -1) {
                        printf("Terminating...\n");
                        terminate = true;
                        goto send_rdma_write;
                    }

                    /* Step 2: send RDMA Read to client to read the input */
                    input_read_done_count[wc.wr_id] = 0;
                    post_rdma_read(
                        img_target,                 // local_src
                        req->input_target_length,   // len
                        mr_images_target->lkey,     // lkey
                        req->input_target_addr,     // remote_dst
                        req->input_target_rkey,     // rkey
                        wc.wr_id);                  // wr_id

                    post_rdma_read(
                        img_reference,              // local_src
                        req->input_reference_length,// len
                        mr_images_reference->lkey,  // lkey
                        req->input_reference_addr,  // remote_dst
                        req->input_reference_rkey,  // rkey
                        wc.wr_id);                  // wr_id
                break;

                case IBV_WC_RDMA_READ:
                    /* Completed RDMA read for a request */
                    input_read_done_count[wc.wr_id]++;
                    if (input_read_done_count[wc.wr_id] == 2){
                        req = &requests[wc.wr_id];
                        img_target = &images_target[wc.wr_id * IMG_BYTES];
                        img_reference = &images_reference[wc.wr_id * IMG_BYTES];
                        img_out = &images_out[wc.wr_id * IMG_BYTES];

                        // Step 3: Process on GPU
                        while(!gpu_context->enqueue(wc.wr_id, img_target, img_reference, img_out)){};
                    }
		        break;
                    
                case IBV_WC_RDMA_WRITE:
                    /* Completed RDMA Write - reuse buffers for receiving the next requests */
                    post_recv(wc.wr_id % OUTSTANDING_REQUESTS);

                    if (terminate)
                        got_last_cqe = true;

                break;
                default:
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }

            // Dequeue completed GPU tasks
            int dequeued_job_id;
            if (gpu_context->dequeue(&dequeued_job_id)) {
                req = &requests[dequeued_job_id];
                img_out = &images_out[dequeued_job_id * IMG_BYTES];

send_rdma_write:
                // Step 4: Send RDMA Write with immediate to client with the response
		        post_rdma_write(
                    req->output_addr,                       // remote_dst
                    terminate ? 0 : req->output_length,     // len
                    req->output_rkey,                       // rkey
                    terminate ? 0 : img_out,                // local_src
                    mr_images_out->lkey,                    // lkey
                    dequeued_job_id + OUTSTANDING_REQUESTS, // wr_id
                    (uint32_t*)&req->request_id);           // immediate
            }
        }
    }
};

class client_rpc_context : public rdma_client_context {
private:
    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

    struct ibv_mr *mr_images_target, *mr_images_reference; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
public:
    explicit client_rpc_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
    }

    ~client_rpc_context()
    {
        kill();
    }

    virtual void set_input_images(uchar *images_target, uchar* images_reference, size_t bytes) override
    {
        /* register a memory region for the input images. */
        mr_images_target = ibv_reg_mr(pd, images_target, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_target) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
        mr_images_reference = ibv_reg_mr(pd, images_reference, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_reference) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        if ((requests_sent - send_cqes_received) == OUTSTANDING_REQUESTS)
            return false;

        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        /* step 1: send request to server using Send operation */
        
        struct rpc_request *req = &requests[requests_sent % OUTSTANDING_REQUESTS];
        req->request_id = job_id;
        req->input_target_rkey = target ? mr_images_target->rkey : 0;
        req->input_target_addr = (uintptr_t)target;
        req->input_target_length = IMG_BYTES;
        req->input_reference_rkey = reference ? mr_images_reference->rkey : 0;
        req->input_reference_addr = (uintptr_t)reference;
        req->input_reference_length = IMG_BYTES;
        req->output_rkey = result ? mr_images_out->rkey : 0;
        req->output_addr = (uintptr_t)result;
        req->output_length = IMG_BYTES;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = job_id; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }

        ++requests_sent;

        return true;
    }

    virtual bool dequeue(int *job_id) override
    {
        /* When WQE is completed we expect a CQE */
        /* We also expect a completion of the RDMA Write with immediate operation from the server to us */
        /* The order between the two is not guarenteed */

        struct ibv_wc wc; /* CQE */
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        if (ncqes == 0)
            return false;

	    VERBS_WC_CHECK(wc);

        switch (wc.opcode) {
        case IBV_WC_SEND:
            ++send_cqes_received;
            return false;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            *job_id = wc.imm_data;
            break;
        default:
            printf("Unexpected completion type\n");
            assert(0);
        }

        /* step 2: po RPC call (st receive buffer for the nextnext RDMA write with imm) */
        post_recv();

        return true;
    }

    void kill()
    {
        while (!enqueue(-1, // Indicate termination
                       NULL, NULL, NULL)) ;
        int job_id = 0;
        bool dequeued;
        do {
            dequeued = dequeue(&job_id);
        } while (!dequeued || job_id != -1);
    }
};

struct Info {
    int number_of_queues;
    void* ci_addr;
    void* pi_addr;
    int gpu_to_cpu_rkey;
    int cpu_to_gpu_rkey;
};
class server_queues_context : public rdma_server_context {
private:
    queue_server gpu_context;
    struct ibv_mr* mr_cpu_to_gpu;
    struct ibv_mr* mr_gpu_to_cpu;
    /* TODO: add memory region(s) for CPU-GPU queues */

public:
    explicit server_queues_context(uint16_t tcp_port) :
        rdma_server_context(tcp_port),
        gpu_context(256)
    {
        /* TODO Initialize additional server MRs as needed. */
        auto my_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
        Info my_info;

        mr_cpu_to_gpu =
            ibv_reg_mr(pd, gpu_context.cpu_to_gpu_queues, sizeof(queue<cpu_to_gpu_queues_entry>), my_flags);
        mr_gpu_to_cpu =
            ibv_reg_mr(pd, gpu_context.gpu_to_cpu_queues, sizeof(queue<gpu_to_cpu_queues_entry>), my_flags);
        
        assert(mr_cpu_to_gpu);
        assert(mr_gpu_to_cpu);

        my_info.number_of_queues = gpu_context.blocks;
        my_info.gpu_to_cpu_rkey = mr_gpu_to_cpu->rkey;
        my_info.cpu_to_gpu_rkey = mr_cpu_to_gpu->rkey;

        my_info.ci_addr = &(gpu_context.cpu_to_gpu_queues[0].ci); //todo: change to offset;
        my_info.pi_addr = &(gpu_context.gpu_to_cpu_queues[0].pi);
        send_over_socket(&my_info, sizeof(Info));
        /* TODO Exchange rkeys, addresses, and necessary information (e.g.
         * number of queues) with the client */

    }

    ~server_queues_context()
    {
        /* TODO destroy the additional server MRs here */
        ibv_dereg_mr(mr_cpu_to_gpu);
        ibv_dereg_mr(mr_gpu_to_cpu);
    }
    bool terminate = false;
    
    virtual void event_loop() override
    {
        /* TODO simplified version of server_rpc_context::event_loop. As the
         * client use one sided operations, we only need one kind of message to
         * terminate the server at the end. */
        while (!terminate) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		        VERBS_WC_CHECK(wc);
                if (wc.opcode == IBV_WC_TM_DEL) {
                    /* Received a new request from the client */
                    printf("Terminating...\n");
                    terminate = true;
                    this->~server_queues_context();
                }
            }
        }
    }
};

class client_queues_context : public rdma_client_context {
private:
    int number_of_queues;
    int consumer_index;
    int producer_index;
    void* ci_remote_addr;
    void* pi_remote_addr;
    struct ibv_mr* mr_indexes;
    struct ibv_mr* mr_images_target; /* Memory region for input images */
    struct ibv_mr* mr_images_reference; /* Memory region for input images */
    struct ibv_mr* mr_images_out; /* Memory region for output images */
    
    int cpu_to_gpu_rkey;
    int gpu_to_cpu_rkey;

    queue<cpu_to_gpu_queues_entry>* cpu_to_gpu_queues;
    queue<gpu_to_cpu_queues_entry>* gpu_to_cpu_queues;
    /* TODO define other memory regions used by the client here */

public:
    client_queues_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
        /* TODO communicate with server to discover number of queues, necessary
         * rkeys / address, or other additional information needed to operate
         * the GPU queues remotely. */
        mr_indexes = ibv_reg_mr(pd, &(consumer_index), sizeof(int)*2, my_flags);

        Info server_info;
        recv_over_socket(&server_info, sizeof(Info));
        number_of_queues = server_info.number_of_queues;
        cpu_to_gpu_rkey = server_info.cpu_to_gpu_rkey;
        gpu_to_cpu_rkey = server_info.gpu_to_cpu_rkey;
        ci_remote_addr = server_info.ci_addr;
        pi_remote_addr = server_info.pi_addr;

        assert(pi_remote_addr);
        assert(ci_remote_addr);
    }

    ~client_queues_context()
    {
	/* TODO terminate the server and release memory regions and other resources */
        ibv_dereg_mr(mr_indexes);
        ibv_dereg_mr(mr_images_target);
        ibv_dereg_mr(mr_images_reference);
        ibv_dereg_mr(mr_images_out);
    }

    virtual void set_input_images(uchar *images_target, uchar* images_reference, size_t bytes) override
    {
        // TODO register memory
        mr_images_target = ibv_reg_mr(pd, images_target, bytes, my_flags);
        if (!mr_images_target) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
        mr_images_reference = ibv_reg_mr(pd, images_reference, bytes, my_flags);
        if (!mr_images_reference) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        // TODO register memory
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, my_flags);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        /* TODO use RDMA Write and RDMA Read operations to enqueue the task on
         * a CPU-GPU producer consumer queue running on the server. */
        /* Create Send Work Request for RDMA Read */
        
        post_rdma_read(
            &consumer_index,                 // local_src
            sizeof(consumer_index),   // len
            mr_indexes->lkey,            // lkey
            (uintptr_t) ci_remote_addr,     // remote_dst
            cpu_to_gpu_rkey,     // rkey
            job_id);                    // wr_id

        printf("did write, job_id: %d\n", job_id);

        printf("consumer_index %d\n", consumer_index);
        printf("producer_index %d\n", producer_index);

        assert(consumer_index < 16);
        assert(producer_index < 16);
        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* TODO use RDMA Write and RDMA Read operations to detect the completion and dequeue a processed image
         * through a CPU-GPU producer consumer queue running on the server. */
        return false;
    }
};

std::unique_ptr<rdma_server_context> create_server(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<server_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<server_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}

std::unique_ptr<rdma_client_context> create_client(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<client_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<client_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}
