# Copyright (c) 2020 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

require "./libcl"

module Cl
  extend self

  def check(errcode : Int32)
    if errcode != 0
      raise "OpenCL Raised #{errcode}"
    end
  end

  def platform_name(id : LibCL::ClPlatformId) : String
    check LibCL.cl_get_platform_info(id, LibCL::ClPlatformInfo::PLATFORM_NAME, 0, nil, out size)
    result = Slice(UInt8).new(size)
    check LibCL.cl_get_platform_info(id, LibCL::ClPlatformInfo::PLATFORM_NAME, size, result, nil)
    String.new(result.to_unsafe)
  end

  def device_name(id : LibCL::ClDeviceId) : String
    check LibCL.cl_get_device_info(id, LibCL::ClDeviceInfo::DEVICE_NAME, 0, nil, out size)
    result = Slice(UInt8).new(size)
    check LibCL.cl_get_device_info(id, LibCL::ClDeviceInfo::DEVICE_NAME, size, result, nil)
    String.new(result.to_unsafe)
  end

  def max_work_groups(id : LibCL::ClDeviceId) : UInt64
    result = 0_u64
    check LibCL.cl_get_device_info(id, LibCL::ClDeviceInfo::DEVICE_MAX_WORK_GROUP_SIZE, sizeof(UInt64), pointerof(result), nil)
    result
  end

  def local_memory(id : LibCL::ClDeviceId) : UInt64
    result = 0_u64
    check LibCL.cl_get_device_info(id, LibCL::ClDeviceInfo::DEVICE_LOCAL_MEM_SIZE, sizeof(UInt64), pointerof(result), nil)
    result
  end

  def global_memory(id : LibCL::ClDeviceId) : UInt64
    result = 0_u64
    check LibCL.cl_get_device_info(id, LibCL::ClDeviceInfo::DEVICE_GLOBAL_MEM_SIZE, sizeof(UInt64), pointerof(result), nil)
    result
  end

  def max_work_items(id : LibCL::ClDeviceId) : Array(UInt64)
    dims = uninitialized UInt32
    check LibCL.cl_get_device_info(id, LibCL::ClDeviceInfo::DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(UInt32), pointerof(dims), nil)
    result = (0...dims).to_a
    check LibCL.cl_get_device_info(id, LibCL::ClDeviceInfo::DEVICE_MAX_WORK_ITEM_SIZES, dims * sizeof(UInt64), result, nil)
    result
  end

  def version(id : LibCL::ClPlatformId) : String
    check LibCL.cl_get_platform_info(id, LibCL::ClPlatformInfo::PLATFORM_VERSION, 0, nil, out size)
    result = Slice(UInt8).new(size)
    check LibCL.cl_get_platform_info(id, LibCL::ClPlatformInfo::PLATFORM_VERSION, size, result, nil)
    String.new(result.to_unsafe)
  end

  def get_platform_by_name(name : String)
    check LibCL.cl_get_platform_ids(0, nil, out num_platforms)
    platforms = Array(LibCL::ClPlatformId).new(num_platforms, LibCL::ClPlatformId.null)
    check LibCL.cl_get_platform_ids(num_platforms, platforms.to_unsafe, nil)

    platforms.each do |platform|
      if platform_name(platform) == name
        return platform
      end
    end

    raise "Platform not found"
  end

  def first_platform
    check LibCL.cl_get_platform_ids(0, nil, out num_platforms)
    if num_platforms == 0
      raise "No platforms found"
    end

    platforms = Array(LibCL::ClPlatformId).new(num_platforms, LibCL::ClPlatformId.null)
    check LibCL.cl_get_platform_ids(num_platforms, platforms.to_unsafe, nil)
    platforms[0]
  end

  def get_devices(platform : LibCL::ClPlatformId) : Array(LibCL::ClDeviceId)
    check LibCL.cl_get_device_ids(platform, LibCL::CL_DEVICE_TYPE_ALL, 0, nil, out num_devices)
    if num_devices == 0
      raise "No devices found"
    end

    devices = Array(LibCL::ClDeviceId).new(num_devices, LibCL::ClDeviceId.null)
    check LibCL.cl_get_device_ids(platform, LibCL::CL_DEVICE_TYPE_ALL, num_devices, devices.to_unsafe, nil)
    devices
  end

  def get_devices(platform : LibCL::ClPlatformId, device_type) : Array(LibCL::ClDeviceId)
    check LibCL.cl_get_device_ids(platform, device_type, 0, nil, out num_devices)
    if num_devices == 0
      raise "No devices found"
    end

    devices = Array(LibCL::ClDeviceId).new(num_devices, LibCL::ClDeviceId.null)
    check LibCL.cl_get_device_ids(platform, device_type, num_devices, devices.to_unsafe, nil)
    devices
  end

  def create_context(devices : Array(LibCL::ClDeviceId)) : LibCL::ClContext
    context = LibCL.cl_create_context(nil, UInt32.new(devices.size), devices.to_unsafe, nil, nil, out status)
    check status
    context
  end

  def command_queue_for(context : LibCL::ClContext, device : LibCL::ClDeviceId) : LibCL::ClCommandQueue
    queue = LibCL.cl_create_command_queue(context, device, 0, out status)
    check status
    queue
  end

  # Creates a command queue with profiling enabled.
  def command_queue_with_profiling(context : LibCL::ClContext, device : LibCL::ClDeviceId) : LibCL::ClCommandQueue
    queue = LibCL.cl_create_command_queue(context, device, LibCL::CL_QUEUE_PROFILING_ENABLE.to_i32, out status)
    check status
    queue
  end

  def opencl_defaults : {Array(LibCL::ClDeviceId), LibCL::ClContext}
    platform = first_platform
    devices = get_devices(platform)
    context = create_context(devices)
    {devices, context}
  end

  def single_device_defaults : {LibCL::ClDeviceId, LibCL::ClContext, LibCL::ClCommandQueue}
    platform = first_platform
    device = get_devices(platform)[0]
    context = create_context([device])
    queue = command_queue_for(context, device)
    {device, context, queue}
  end

  def first_gpu_defaults : {LibCL::ClDeviceId, LibCL::ClContext, LibCL::ClCommandQueue}
    platform = first_platform
    device = get_devices(platform, LibCL::CL_DEVICE_TYPE_GPU)[0]
    context = create_context([device])
    queue = command_queue_for(context, device)
    {device, context, queue}
  end

  def create_program(context : LibCL::ClContext, body : String) : LibCL::ClProgram
    lines = [body.to_unsafe]
    result = LibCL.cl_create_program_with_source(context, 1, lines.to_unsafe, nil, out status)
    check status
    result
  end

  def create_program_binary(context : LibCL::ClContext, device : LibCL::ClDeviceId, body : String) : LibCL::ClProgram
    lines = [body.to_unsafe]
    l = body.size.to_u64
    result = LibCL.cl_create_program_with_binary(context, 1, pointerof(device), pointerof(l), lines.to_unsafe, out binary_status, out status)
    check status
    result
  end

  def build_on(program : LibCL::ClProgram, devices : Array(LibCL::ClDeviceId))
    LibCL.cl_build_program(program, UInt32.new(devices.size), devices.to_unsafe, nil, nil, nil)
  end

  def build_on(program : LibCL::ClProgram, device : LibCL::ClDeviceId)
    build_on(program, [device])
  end

  def create_and_build(context : LibCL::ClContext, body : String, device : LibCL::ClDeviceId) : LibCL::ClProgram
    result = create_program(context, body)
    build_on(result, device)
    result
  end

  def create_and_build(context : LibCL::ClContext, body : String, devices : Array(LibCL::ClDeviceId)) : LibCL::ClProgram
    result = create_program(context, body)
    build_on(result, devices)
    result
  end

  def create_and_build_binary(context : LibCL::ClContext, body : String, device : LibCL::ClDeviceId) : LibCL::ClProgram
    result = create_program_binary(context, device, body)
    build_on(result, device)
    result
  end

  def buffer(context : LibCL::ClContext, size : UInt64, flags : LibCL::ClMemFlags = LibCL::ClMemFlags::READ_WRITE, dtype : U.class = Float64) : LibCL::ClMem forall U
    buffer = LibCL.cl_create_buffer(context, flags, size * sizeof(U), nil, out status)
    check status
    buffer
  end

  def buffer_like(context : LibCL::ClContext, xs : Array(U), flags : LibCL::ClMemFlags = LibCL::ClMemFlags::READ_WRITE) : LibCL::ClMem forall U
    buffer(context, UInt64.new(xs.size), flags, dtype: U)
  end

  def build_errors(program : LibCL::ClProgram, devices : Array(LibCL::ClDeviceId)) : String
    check LibCL.cl_get_program_build_info(program, devices[0], LibCL::ClProgramBuildInfo::PROGRAM_BUILD_LOG, 0, nil, out log_size)
    result = Bytes.new(log_size)
    check LibCL.cl_get_program_build_info(program, devices[0], LibCL::ClProgramBuildInfo::PROGRAM_BUILD_LOG, log_size, result, nil)
    String.new(result.to_unsafe)
  end

  def create_kernel(program : LibCL::ClProgram, name : String) : LibCL::ClKernel
    result = LibCL.cl_create_kernel(program, name.to_unsafe, out status)
    check status
    result
  end

  def set_arg(kernel : LibCL::ClKernel, item : LibCL::ClMem, index : UInt32)
    check LibCL.cl_set_kernel_arg(kernel, index, sizeof(LibCL::ClMem), pointerof(item))
  end

  def set_arg(kernel : LibCL::ClKernel, item, index : UInt32)
    check LibCL.cl_set_kernel_arg(kernel, index, sizeof(typeof(item)).to_u64, pointerof(item).as(Void*))
  end

  def args(kernel : LibCL::ClKernel, *args)
    args.each_with_index do |arg, i|
      set_arg(kernel, arg, UInt32.new(i))
    end
  end

  def run(queue : LibCL::ClCommandQueue, kernel : LibCL::ClKernel, work : Int)
    run_with_event(queue, kernel, work)
  end

  def run_with_event(queue : LibCL::ClCommandQueue, kernel : LibCL::ClKernel, work : Int) : LibCL::ClEvent
    global_work_size = [UInt64.new(work), 0_u64, 0_u64]
    check LibCL.cl_enqueue_nd_range_kernel(queue, kernel, 1, nil, global_work_size.to_unsafe, nil, 0, nil, out event)
    event
  end

  def run(queue : LibCL::ClCommandQueue, kernel : LibCL::ClKernel, total_work : Int, local_work : Int)
    global_work_size = [UInt64.new(total_work), 0_u64, 0_u64]
    local_work_size = [UInt64.new(local_work), 0_u64, 0_u64]
    check LibCL.cl_enqueue_nd_range_kernel(queue, kernel, 1, nil, global_work_size.to_unsafe, local_work_size.to_unsafe, 0, nil, nil)
  end

  def run2d(queue : LibCL::ClCommandQueue, kernel : LibCL::ClKernel, total_work : Tuple(Int, Int))
    a, b = total_work
    global_work_size = [UInt64.new(a), UInt64.new(b), 0_u64]
    check LibCL.cl_enqueue_nd_range_kernel(queue, kernel, 2, nil, global_work_size.to_unsafe, nil, 0, nil, nil)
  end

  def run2d(queue : LibCL::ClCommandQueue, kernel : LibCL::ClKernel, total_work : Tuple(Int, Int), local_work : Tuple(Int, Int))
    a, b = total_work
    c, d = local_work
    global_work_size = [UInt64.new(a), UInt64.new(b), 0_u64]
    local_work_size = [UInt64.new(c), UInt64.new(d), 0_u64]
    check LibCL.cl_enqueue_nd_range_kernel(queue, kernel, 2, nil, global_work_size.to_unsafe, local_work_size.to_unsafe, 0, nil, nil)
  end

  def run3d(queue : LibCL::ClCommandQueue, kernel : LibCL::ClKernel, total_work : Tuple(Int, Int, Int))
    global_work_size = [UInt64.new(total_work[0]), UInt64.new(total_work[1]), UInt64.new(total_work[2])]
    check LibCL.cl_enqueue_nd_range_kernel(queue, kernel, 3, nil, global_work_size.to_unsafe, nil, 0, nil, nil)
  end

  def run3d(queue : LibCL::ClCommandQueue, kernel : LibCL::ClKernel, total_work : Tuple(Int, Int, Int), local_work : Tuple(Int, Int, Int))
    global_work_size = [UInt64.new(total_work[0]), UInt64.new(total_work[1]), UInt64.new(total_work[2])]
    local_work_size = [UInt64.new(local_work[0]), UInt64.new(local_work[1]), UInt64.new(local_work[2])]
    check LibCL.cl_enqueue_nd_range_kernel(queue, kernel, 3, nil, global_work_size.to_unsafe, local_work_size.to_unsafe, 0, nil, nil)
  end

  def write(queue : LibCL::ClCommandQueue, src : Pointer(U), dest : LibCL::ClMem, size : UInt64) forall U
    check LibCL.cl_enqueue_write_buffer(queue, dest, LibCL::CL_FALSE, 0, size, src, 0, nil, nil)
  end

  def write(queue : LibCL::ClCommandQueue, src : Array(U), dest : LibCL::ClMem) forall U
    write(queue, src.to_unsafe, dest, UInt64.new(src.size * sizeof(U)))
  end

  def fill(queue : LibCL::ClCommandQueue, buffer : LibCL::ClMem, value : U, size : UInt64) forall U
    LibCL.cl_enqueue_fill_buffer(queue, buffer, pointerof(value), sizeof(U), 0, size * sizeof(U), 0, nil, nil)
  end

  def read(queue : LibCL::ClCommandQueue, dest : Pointer(U), src : LibCL::ClMem, size : Int) forall U
    LibCL.cl_enqueue_read_buffer(queue, src, LibCL::CL_TRUE, 0, size, dest, 0, nil, nil)
  end

  def read(queue : LibCL::ClCommandQueue, dest : Array(U), src : LibCL::ClMem) forall U
    read(queue, dest.to_unsafe, src, UInt64.new(dest.size * sizeof(U)))
  end

  def release_buffer(buffer : LibCL::ClMem)
    check LibCL.cl_release_mem_object(buffer)
  end

  def release_queue(queue : LibCL::ClCommandQueue)
    check LibCL.cl_release_queue(queue)
  end

  def release_context(context : LibCL::ClContext)
    check LibCL.cl_release_context(context)
  end

  def release_kernel(kernel : LibCL::ClKernel)
    check LibCL.cl_release_kernel(kernel)
  end

  def release_program(program : LibCL::ClProgram)
    check LibCL.cl_release_program(program)
  end

  def release_event(event : LibCL::ClEvent)
    check LibCL.cl_release_event(event)
  end

  # Returns a UInt64 timestamp (in nanoseconds) for a profiling point of an event.
  # The *param* should be one of `LibCL::ClProfilingInfo::PROFILING_COMMAND_START`, etc.
  def get_profiling_info(event : LibCL::ClEvent, param : LibCL::ClProfilingInfo) : UInt64
    result = 0_u64
    check LibCL.cl_get_event_profiling_info(event, param, sizeof(UInt64), pointerof(result).as(Void*), nil)
    result
  end

  def create_user_event(context : LibCL::ClContext) : LibCL::ClEvent
    event = LibCL.cl_create_user_event(context, out status)
    check status
    event
  end

  def set_user_event_status(event : LibCL::ClEvent, status : Int32)
    check LibCL.cl_set_user_event_status(event, status)
  end

  # ---------------------------------------------------------------------------
  # Advanced OpenCL Features (2.0+)
  # ---------------------------------------------------------------------------

  # Returns true if the device supports SPIR-V Intermediate Language (OpenCL 2.1+)
  def supports_il?(device : LibCL::ClDeviceId) : Bool
    LibCL.cl_get_device_info(device, LibCL::ClDeviceInfo::DEVICE_IL_VERSION, 0, nil, out size)
    size > 0
  rescue
    false
  end

  # Creates a program from SPIR-V Intermediate Language bytes
  def create_program_with_il(context : LibCL::ClContext, il_bytes : Bytes) : LibCL::ClProgram
    result = LibCL.cl_create_program_with_il(
      context,
      il_bytes.to_unsafe.as(Void*),
      LibC::SizeT.new(il_bytes.size),
      out status
    )
    check status
    result
  end

  # Returns true if the device supports Pipes (OpenCL 2.0+)
  def supports_pipes?(device : LibCL::ClDeviceId) : Bool
    max_args = 0_u32
    status = LibCL.cl_get_device_info(
      device,
      LibCL::ClDeviceInfo::DEVICE_MAX_PIPE_ARGS,
      sizeof(UInt32),
      pointerof(max_args).as(Void*),
      nil
    )
    status == 0 && max_args > 0
  rescue
    false
  end

  # Creates a Pipe memory object
  def create_pipe(
    context : LibCL::ClContext,
    flags : LibCL::ClMemFlags,
    packet_size : UInt32,
    max_packets : UInt32
  ) : LibCL::ClMem
    result = LibCL.cl_create_pipe(
      context,
      flags,
      packet_size,
      max_packets,
      nil,
      out status
    )
    check status
    result
  end

  # Returns true if the given OpenCL device supports Fine-Grained buffer SVM.
  def supports_fine_grain_svm?(device : LibCL::ClDeviceId) : Bool
    caps = 0_u64
    status = LibCL.cl_get_device_info(
      device,
      LibCL::ClDeviceInfo.new(LibCL::CL_DEVICE_SVM_CAPABILITIES.to_i32),
      sizeof(UInt64),
      pointerof(caps).as(Void*),
      nil
    )
    # CL_DEVICE_SVM_FINE_GRAIN_BUFFER is bit 1 (value 2)
    status == 0 && (caps & 2_u64) != 0
  rescue
    false
  end

  # Returns true if the device supports the Command Buffer extension (cl_khr_command_buffer)
  def supports_command_buffers?(device : LibCL::ClDeviceId) : Bool
    check LibCL.cl_get_device_info(device, LibCL::ClDeviceInfo::DEVICE_EXTENSIONS, 0, nil, out size)
    result = Slice(UInt8).new(size)
    check LibCL.cl_get_device_info(device, LibCL::ClDeviceInfo::DEVICE_EXTENSIONS, size, result, nil)
    String.new(result.to_unsafe).includes?("cl_khr_command_buffer")
  rescue
    false
  end

  private def get_extension_func(device : LibCL::ClDeviceId, name : String)
    platform = LibCL::ClPlatformId.null
    check LibCL.cl_get_device_info(device, LibCL::ClDeviceInfo::DEVICE_PLATFORM, sizeof(LibCL::ClPlatformId), pointerof(platform).as(Void*), nil)
    addr = LibCL.cl_get_extension_function_address_for_platform(platform, name)
    raise "Extension function #{name} not found" if addr.null?
    addr
  end

  # ---------------------------------------------------------------------------
  # OpenCL 2.0+ — Shared Virtual Memory helpers
  # ---------------------------------------------------------------------------

  # Wraps a raw SVM (Shared Virtual Memory) pointer returned by `clSVMAlloc`.
  # The pointer can be accessed directly by both host and device without explicit
  # copy operations when the device supports coarse-grain buffer SVM.
  class SVMPointer
    getter raw : Void*

    def initialize(@raw : Void*)
    end
  end

  # Returns `true` if the given OpenCL device supports at least coarse-grain
  # buffer Shared Virtual Memory (SVM capabilities bitmask > 0).
  def svm_supported?(device : LibCL::ClDeviceId) : Bool
    caps = 0_u64
    status = LibCL.cl_get_device_info(
      device,
      LibCL::ClDeviceInfo.new(LibCL::CL_DEVICE_SVM_CAPABILITIES),
      sizeof(UInt64),
      pointerof(caps).as(Void*),
      nil
    )
    status == 0 && caps > 0
  rescue
    false
  end

  # Maps an SVM region into host-accessible memory synchronously or
  # asynchronously depending on *blocking*.
  #
  # * *queue*    – the OpenCL command queue
  # * *blocking* – `LibCL::CL_TRUE` blocks until the map operation completes
  # * *flags*    – `LibCL::ClMapFlags` (READ, WRITE, WRITE_INVALIDATE_REGION)
  # * *ptr*      – the SVM `Void*` to map (must have been allocated via `clSVMAlloc`)
  # * *size*     – number of bytes to map
  def map_svm(
    queue    : LibCL::ClCommandQueue,
    blocking : Int32,
    flags    : UInt64,
    ptr      : Void*,
    size     : UInt64
  )
    rc = LibCL.cl_enqueue_svm_map(queue, blocking, flags, ptr, size, 0_u32, nil, nil)
    check rc
  end

  # Unmaps a previously mapped SVM region, signalling to the driver that
  # host writes are complete and the device may resume accessing the memory.
  def unmap_svm(queue : LibCL::ClCommandQueue, ptr : Void*)
    rc = LibCL.cl_enqueue_svm_unmap(queue, ptr, 0_u32, nil, nil)
    check rc
  end

  # ---------------------------------------------------------------------------
  # OpenCL 1.1+ — Sub-buffer helper
  # ---------------------------------------------------------------------------

  # Creates an OpenCL sub-buffer that aliases a byte range of *buffer*.
  #
  # The *byte_offset* must be aligned to the device's `CL_DEVICE_MEM_BASE_ADDR_ALIGN`
  # value (retrievable via `Cl.mem_base_addr_align`).
  def create_sub_buffer(
    buffer      : LibCL::ClMem,
    byte_offset : UInt64,
    byte_size   : UInt64
  ) : LibCL::ClMem
    region = LibCL::ClBufferRegion.new(origin: byte_offset, size: byte_size)
    status = 0
    sub_buf = LibCL.cl_create_sub_buffer(
      buffer,
      LibCL::ClMemFlags::READ_WRITE,
      LibCL::CL_BUFFER_CREATE_TYPE_REGION,
      pointerof(region).as(Void*),
      pointerof(status)
    )
    check status
    sub_buf
  end

  # Returns the base address alignment (in bits) required for sub-buffers on
  # the given device. Divide by 8 to get byte alignment.
  def mem_base_addr_align(device : LibCL::ClDeviceId) : UInt32
    result = 0_u32
    check LibCL.cl_get_device_info(
      device,
      LibCL::ClDeviceInfo::DEVICE_MEM_BASE_ADDR_ALIGN,
      sizeof(UInt32),
      pointerof(result).as(Void*),
      nil
    )
    result
  end
end
