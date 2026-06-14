require "../src/opencl"

# SVM (Shared Virtual Memory) allows host and device to share a single virtual address space.
# This eliminates the need for explicit buffer copies on supported hardware.

device, context, queue = Cl.single_device_defaults

unless Cl.svm_supported?(device)
  puts "SVM is not supported on this device. Skipping example."
  exit
end

puts "Device supports SVM: #{Cl.device_name(device)}"

# Allocate 1024 floats in SVM
size = 1024_u64 * sizeof(Float32)
# clSVMAlloc returns a raw pointer accessible by both CPU and GPU
ptr = LibCL.cl_svm_alloc(context, LibCL::ClMemFlags::READ_WRITE.to_u64, size, 0_u32)

if ptr.null?
  puts "Failed to allocate SVM memory"
  exit
end

begin
  # Map the memory to make it visible to the CPU
  Cl.map_svm(queue, LibCL::CL_TRUE, LibCL::ClMapFlags::WRITE.to_u64, ptr, size)
  
  # Initialize data using Crystal's pointer API
  float_ptr = ptr.as(Float32*)
  1024.times { |i| float_ptr[i] = i.to_f32 }
  
  puts "SVM initialization: [#{float_ptr[0]}, #{float_ptr[1]}, #{float_ptr[2]}]"

  # On some devices (like POCL), svm_alloc might return success but kernel execution
  # on SVM pointers might require specific extensions or flags.
  # We will just verify that the pointer is usable from the host.
  
  Cl.unmap_svm(queue, ptr)
  puts "SVM memory allocated and used successfully on Host."
ensure
  LibCL.cl_svm_free(context, ptr)
end

Cl.release_queue(queue)
Cl.release_context(context)
