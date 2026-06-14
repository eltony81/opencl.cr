# opencl.cr

Crystal bindings for OpenCL, providing utilities and high/low-level API wrappers for Crystal applications.

This library is primarily maintained to provide necessary utilites to the
`num.cr` numerical library, so not all features may be covered.  This library
should however cover all basic use cases, as well as provide a lower level ability
to implement more advanced use cases.  Feel free to submit PR's to add functionality

## Installation

1. Add the dependency to your `shard.yml`:

   ```yaml
   dependencies:
     opencl:
       github: eltony81/opencl.cr
   ```

2. Run `shards install`

## Key Enhancements in this Fork

This fork significantly extends the original library with support for modern OpenCL (2.x/3.0) features, improved safety, and better performance analysis:

### 🚀 Modern OpenCL Features (2.0+)
- **Shared Virtual Memory (SVM)**: Support for `clSVMAlloc`, mapping, and unmapping. Allows host and device to share memory regions more efficiently.
- **SPIR-V / Intermediate Language**: Support for loading pre-compiled SPIR-V binaries via `Cl.create_program_with_il`.
- **Pipes**: Hardware-level FIFO queues for kernel-to-kernel communication.
- **Sub-buffers**: Ability to create memory aliases for specific byte ranges of an existing buffer.

### 📊 Profiling & Performance
- **Kernel Profiling**: Enable profiling on command queues and retrieve nanosecond-precision execution timestamps (Queued, Submit, Start, End).
- **Command Buffers**: Low-level bindings and extension support for `cl_khr_command_buffer`, reducing CPU overhead for batch operations.

### 🛠️ Developer Experience & Safety
- **Hardware Capability Queries**: New methods like `Cl.supports_il?`, `Cl.supports_pipes?`, and `Cl.supports_fine_grain_svm?` to gracefully handle different hardware capabilities.
- **User Events**: Full support for creating and triggering manual events from the host.
- **Improved FFI Safety**: 
    - Automatic stripping of null terminators from C-strings (Platform/Device names, versions, build logs).
    - Generic `Cl.set_arg` that supports any value type automatically.
    - Safer memory management for platform and device IDs using Crystal-managed arrays.
    - Explicit usage of `.to_unsafe` for better clarity and FFI compatibility.

### 🎨 Graphics Interop
- **OpenGL Interop**: Initial bindings for sharing buffers between OpenCL and OpenGL (`clCreateFromGLBuffer`).

## Usage

```crystal
require "opencl"

device, context, queue = Cl.single_device_defaults
puts Cl.device_name(device)
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
