///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#include <infiniband/verbs.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <string.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "ex3.h"
#include "randomize_images.h"
#include "ex3-cpu.h"

#include <random>

#define SQR(a) ((a) * (a))

long long int distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    long long int distance_sqr = 0;
    auto printed = false;
    for (size_t i = 0; i < N_IMAGES * IMG_BYTES; i++) {
        if (img_arr1[i] != img_arr2[i])
            if(!printed) {
                dbg_printf("cpu[0x%4lx/0x%04lx] == 0x%x != 0x%x\n", i / (IMG_BYTES), i % (IMG_BYTES), img_arr1[i], img_arr2[i]);
                printed = true;
            }
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}

int process_images(mode_enum mode, std::unique_ptr<rdma_client_context>& client)
{
    std::unique_ptr<uchar[]> images_target = std::make_unique<uchar[]>(IMG_BYTES * N_IMAGES);
    std::unique_ptr<uchar[]> images_reference = std::make_unique<uchar[]>(IMG_BYTES * N_IMAGES);
    std::unique_ptr<uchar[]> images_out_cpu = std::make_unique<uchar[]>(IMG_BYTES * N_IMAGES);
    std::unique_ptr<uchar[]> images_out_gpu = std::make_unique<uchar[]>(IMG_BYTES * N_IMAGES);

    client->set_input_images(images_target.get(), images_reference.get(), IMG_BYTES * N_IMAGES);
    client->set_output_images(images_out_gpu.get(), IMG_BYTES * N_IMAGES);

    double t_start, t_finish;

    /* instead of loading real images, we'll load the arrays with random data */
    printf("\n=== Randomizing images ===\n");
    t_start = get_time_msec();
    randomizeImages(images_target.get());
    randomizeImages(images_reference.get());
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    // CPU computation. For reference. Do not change
    printf("\n=== CPU ===\n");
    t_start = get_time_msec();
    for (size_t i = 0; i < N_IMAGES; i++) {
        uchar *img_target = &images_target[i * IMG_BYTES];
        uchar *img_reference = &images_reference[i * IMG_BYTES];
        uchar *img_out = &images_out_cpu[i * IMG_BYTES];
        cpu_process(
            reinterpret_cast<uchar(*)[CHANNELS]>(img_target),
            reinterpret_cast<uchar(*)[CHANNELS]>(img_reference),
            reinterpret_cast<uchar(*)[CHANNELS]>(img_out),
            SIZE, SIZE);
    }
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    printf("\n=== Client-Server ===\n");
    printf("mode = %s\n", mode == MODE_RPC_SERVER ? "rpc" : "queue");

    long long int distance_sqr;
    std::vector<double> req_t_start(N_IMAGES, NAN), req_t_end(N_IMAGES, NAN);

    t_start = get_time_msec();
    size_t next_job_id = 0;
    size_t num_dequeued = 0;
    const size_t total_requests = N_IMAGES;// * 3;

    while (next_job_id < total_requests || num_dequeued < total_requests) {
        int dequeued_img_id;
        if (client->dequeue(&dequeued_img_id)) {
            ++num_dequeued;
            req_t_end[dequeued_img_id % N_IMAGES] = get_time_msec();
        }

        /* If we are done with enqueuing, just loop until all are dequeued */
        if (next_job_id == total_requests)
            continue;

        /* Enqueue a new image */
        req_t_start[next_job_id % N_IMAGES] = get_time_msec();
        if (client->enqueue(next_job_id,
            &images_target[(next_job_id % N_IMAGES) * IMG_BYTES],
            &images_reference[(next_job_id % N_IMAGES) * IMG_BYTES],
            &images_out_gpu[(next_job_id % N_IMAGES) * IMG_BYTES]))
        {
            ++next_job_id;
        }
    }
    t_finish = get_time_msec();
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu.get(), images_out_gpu.get());
    printf("distance from baseline %lld (should be zero)\n", distance_sqr);
    printf("throughput = %.1lf (req/sec)\n", total_requests / (t_finish - t_start) * 1e+3);

    print_latency("overall", req_t_start, req_t_end);

    return 0;
}

int main(int argc, char *argv[]) {
    enum mode_enum mode;
    uint16_t tcp_port;

    parse_arguments(argc, argv, &mode, &tcp_port);
    if (!tcp_port) {
        printf("usage: %s <rpc|queue> <tcp port>\n", argv[0]);
        exit(1);
    }

    auto client = create_client(mode, tcp_port);

    process_images(mode, client);

    return 0;
}
