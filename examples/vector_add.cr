require "../src/opencl"

# 1. Setup: Get a device, context, and command queue
# This helper method picks the first platform and first device automatically.
device, context, queue = Cl.single_device_defaults
puts "Using device: #{Cl.device_name(device)}"

# 2. Prepare Data
count = 1024
a = Array(Float32).new(count) { |i| i.to_f32 }
b = Array(Float32).new(count) { |i| i.to_f32 * 2.0_f32 }
results = Array(Float32).new(count, 0.0_f32)

# 3. Create OpenCL Buffers
# Cl.buffer_like creates a buffer with the same size and type as the input array.
buf_a = Cl.buffer_like(context, a, LibCL::ClMemFlags::READ_ONLY)
buf_b = Cl.buffer_like(context, b, LibCL::ClMemFlags::READ_ONLY)
buf_res = Cl.buffer_like(context, results, LibCL::ClMemFlags::WRITE_ONLY)

# 4. Write data to the GPU
Cl.write(queue, a, buf_a)
Cl.write(queue, b, buf_b)

# 5. Compile the Kernel
source = "
__kernel void vector_add(__global const float* a, __global const float* b, __global float* res) {
    int i = get_global_id(0);
    res[i] = a[i] + b[i];
}
"
program = Cl.create_and_build(context, source, device)
kernel = Cl.create_kernel(program, "vector_add")

# 6. Set Arguments and Run
# The Cl.args macro makes it easy to pass multiple arguments at once.
Cl.args(kernel, buf_a, buf_b, buf_res)
Cl.run(queue, kernel, count)

# 7. Read results back to the CPU
Cl.read(queue, results, buf_res)

# 8. Verify
puts "First 5 results: #{results[0...5]}"
puts "Verification: #{results[10] == a[10] + b[10] ? "PASSED" : "FAILED"}"

# 9. Cleanup
Cl.release_kernel(kernel)
Cl.release_program(program)
Cl.release_buffer(buf_a)
Cl.release_buffer(buf_b)
Cl.release_buffer(buf_res)
Cl.release_queue(queue)
Cl.release_context(context)
