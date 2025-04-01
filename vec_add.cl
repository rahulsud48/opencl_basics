__kernel void vec_add(__global int* A, __global int* B, __global int* C, int n) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);
    int num_groups = get_num_groups(0);
    if (global_id < n) {
        C[global_id] = A[global_id] + B[global_id];
    }
}
