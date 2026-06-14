# opencl.cr

[![CI](https://github.com/eltony81/opencl.cr/actions/workflows/ci.yml/badge.svg)](https://github.com/eltony81/opencl.cr/actions/workflows/ci.yml)

Crystal bindings for OpenCL, providing utilities and high/low-level API wrappers for Crystal applications.

This library is primarily maintained to provide necessary utilities to the
`num.cr` numerical library, so not all features may be covered. This library
should however cover all basic use cases, as well as provide a lower level ability
to implement more advanced use cases. Feel free to submit PR's to add functionality.

## Installation

1. Add the dependency to your `shard.yml`:

   ```yaml
   dependencies:
     opencl:
       github: eltony81/opencl.cr
   ```

2. Run `shards install`

## Key Enhancements in this Fork

This fork significantly extends the original library with support for modern OpenCL (2.x/3.0) features, improved safety, and better performance analysis.

### 🆕 New Functions & Methods
Compared to the original repository, this fork adds the following:

#### **Crystal High-Level Wrapper (`Cl` module)**
- **Profiling**: `Cl.command_queue_with_profiling`, `Cl.get_profiling_info`, `Cl.run_with_event`.
- **SVM (Shared Virtual Memory)**: `Cl.svm_supported?`, `Cl.supports_fine_grain_svm?`, `Cl.map_svm`, `Cl.unmap_svm`.
- **Sub-buffers**: `Cl.create_sub_buffer`, `Cl.mem_base_addr_align`.
- **Hardware Queries**: `Cl.supports_il?`, `Cl.supports_pipes?`, `Cl.supports_command_buffers?`.
- **Advanced Programs**: `Cl.create_program_with_il` (SPIR-V support), `Cl.create_pipe`.
- **Control & Sync**: `Cl.create_user_event`, `Cl.set_user_event_status`, `Cl.release_event`, `Cl.finish`.
- **Generic Kernel Handling**: `Cl.args` (macro for safe mixed-type arguments), generic `Cl.set_arg`.

#### **C Bindings (`LibCL` module)**
- **Memory**: `clSVMAlloc`, `clSVMFree`, `clEnqueueSVMMap`, `clEnqueueSVMUnmap`, `clCreateSubBuffer`, `clCreatePipe`.
- **Execution & Events**: `clCreateUserEvent`, `clSetUserEventStatus`, `clGetEventProfilingInfo`, `clFinish`.
- **Extensions**: `clGetExtensionFunctionAddress`, `clGetExtensionFunctionAddressForPlatform`, `clCreateCommandQueueWithProperties`.
- **Programs & Interop**: `clCreateProgramWithIL`, `clCreateFromGLBuffer`.

### 🚀 Modern OpenCL Features (2.0+)
- **Shared Virtual Memory (SVM)**: Allows host and device to share memory regions more efficiently.
- **SPIR-V / Intermediate Language**: Support for loading pre-compiled binaries via `Cl.create_program_with_il`.
- **Pipes**: Hardware-level FIFO queues for kernel-to-kernel communication.
- **Sub-buffers**: Create memory aliases for specific byte ranges.

### 📊 Profiling & Performance
- **Kernel Profiling**: Nanosecond-precision execution timestamps (Queued, Submit, Start, End).
- **Improved FFI Safety**: 
    - Automatic stripping of null terminators from C-strings (Platform/Device names, versions, build logs).
    - Macro-based `Cl.args` to prevent `CL_INVALID_ARG_SIZE` errors with mixed types.
    - Managed memory for platform/device discovery.

## Usage Examples

Here are some of the most important functions and how to use them. For complete scripts, see the `examples/` directory.

### Basic Vector Addition
```crystal
require "opencl"

device, context, queue = Cl.single_device_defaults

# Create buffers from Crystal arrays
a = [1.0_f32, 2.0_f32, 3.0_f32]
buf_a = Cl.buffer_like(context, a)
Cl.write(queue, a, buf_a)

# ... (setup other buffers and build program)

# Set kernel arguments and run
Cl.args(kernel, buf_a, buf_b, buf_res)
Cl.run(queue, kernel, a.size)
```

### Advanced Features

#### Kernel Profiling
Measure execution time in nanoseconds:
```crystal
queue = Cl.command_queue_with_profiling(context, device)
event = Cl.run_with_event(queue, kernel, work_size)

# Wait for completion
LibCL.cl_wait_for_events(1, pointerof(event))

start = Cl.get_profiling_info(event, LibCL::ClProfilingInfo::PROFILING_COMMAND_START)
end_t = Cl.get_profiling_info(event, LibCL::ClProfilingInfo::PROFILING_COMMAND_END)
puts "Execution time: #{end_t - start} ns"
```

#### Shared Virtual Memory (SVM)
Share memory pointers between Host and Device (OpenCL 2.0+):
```crystal
if Cl.svm_supported?(device)
  # Allocate SVM memory
  ptr = LibCL.cl_svm_alloc(context, LibCL::ClMemFlags::READ_WRITE.to_u64, size, 0_u32)
  
  # Map for Host access
  Cl.map_svm(queue, LibCL::CL_TRUE, LibCL::ClMapFlags::WRITE.to_u64, ptr, size)
  # ... use ptr directly ...
  Cl.unmap_svm(queue, ptr)
end
```

#### Multi-dimensional Execution
```crystal
# Run a 2D grid kernel
Cl.run2d(queue, kernel, {width, height})

# Run a 3D grid kernel with local work item size
Cl.run3d(queue, kernel, {100, 100, 100}, {10, 10, 10})
```

## Contributing

1. Fork it (<https://github.com/eltony81/opencl.cr/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- [Chris Zimmerman](https://github.com/christopherzimmerman) - creator
- [eltony81](https://github.com/eltony81) - maintainer
