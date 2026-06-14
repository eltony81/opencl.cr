require "../src/opencl"

# This example demonstrates how to measure kernel execution time using the Profiling API.
device, context, _ = Cl.single_device_defaults

# Create a queue with profiling enabled
queue = Cl.command_queue_with_profiling(context, device)
puts "Profiling enabled on: #{Cl.device_name(device)}"

# A dummy kernel that performs some work
source = "
__kernel void heavy_work(__global float* a, int iterations) {
    int i = get_global_id(0);
    float val = a[i];
    for(int j = 0; j < iterations; j++) {
        val = native_cos(val) + native_sin(val);
    }
    a[i] = val;
}
"
program = Cl.create_and_build(context, source, device)
kernel = Cl.create_kernel(program, "heavy_work")

count = 1024 * 1024
buf = Cl.buffer(context, count.to_u64, dtype: Float32)
Cl.args(kernel, buf, 100_i32)

# Run the kernel and get an event
event = Cl.run_with_event(queue, kernel, count)

# Wait for the kernel to finish
LibCL.cl_wait_for_events(1, pointerof(event))

# Retrieve profiling timestamps (in nanoseconds)
start_time = Cl.get_profiling_info(event, LibCL::ClProfilingInfo::PROFILING_COMMAND_START)
end_time = Cl.get_profiling_info(event, LibCL::ClProfilingInfo::PROFILING_COMMAND_END)

duration_ns = end_time - start_time
puts "Kernel execution took: #{duration_ns / 1_000_000.0} ms"

Cl.release_event(event)
Cl.release_kernel(kernel)
Cl.release_program(program)
Cl.release_buffer(buf)
Cl.release_queue(queue)
Cl.release_context(context)
