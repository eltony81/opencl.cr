require "./spec_helper"

private def has_opencl?
  Cl.first_platform
  true
rescue
  false
end

describe Cl do
  if has_opencl?
    it "can create context and allocate sub-buffers" do
      device, context, queue = Cl.single_device_defaults
      buf = Cl.buffer(context, 128_u64, dtype: Float64)
      begin
        sub_buf = Cl.create_sub_buffer(buf, 0_u64, 512_u64)
        sub_buf.should_not be_nil
        Cl.release_buffer(sub_buf)
      ensure
        Cl.release_buffer(buf)
      end
    end

    it "can get platform by name" do
      platform = Cl.first_platform
      name = Cl.platform_name(platform)

      found_platform = Cl.get_platform_by_name(name)
      Cl.platform_name(found_platform).should eq(name)
    end

    it "can create and set user events" do
      device, context, queue = Cl.single_device_defaults
      event = Cl.create_user_event(context)
      event.should_not be_nil
      Cl.set_user_event_status(event, 0)
      Cl.release_event(event)
    end

    it "can create sub-buffers with alignment if supported" do
      device, context, queue = Cl.single_device_defaults
      align_bits = Cl.mem_base_addr_align(device)
      align_bytes = align_bits / 8

      # We need a buffer big enough to have at least one aligned offset
      buf = Cl.buffer(context, 256_u64, dtype: UInt8)
      begin
        # Use the alignment to pick a safe offset
        offset = align_bytes.to_u64
        size = 64_u64

        sub_buf = Cl.create_sub_buffer(buf, offset, size)
        sub_buf.should_not be_nil
        Cl.release_buffer(sub_buf)
      ensure
        Cl.release_buffer(buf)
      end
    end

    it "can query advanced features (SPIR-V, Pipes, SVM)" do
      device, context, queue = Cl.single_device_defaults

      # We just want to make sure these queries don't raise exceptions
      supports_il = Cl.supports_il?(device)
      supports_pipes = Cl.supports_pipes?(device)
      supports_fg_svm = Cl.supports_fine_grain_svm?(device)

      (supports_il || !supports_il).should be_true
      (supports_pipes || !supports_pipes).should be_true
      (supports_fg_svm || !supports_fg_svm).should be_true
    end

    it "can profile kernel execution" do
      device, context, _ = Cl.single_device_defaults
      queue = Cl.command_queue_with_profiling(context, device)

      program = Cl.create_and_build(context, "__kernel void test(__global float* a) { a[get_global_id(0)] = 1.0; }", device)
      kernel = Cl.create_kernel(program, "test")
      buf = Cl.buffer(context, 1024_u64, dtype: Float32)
      Cl.args(kernel, buf)

      event = Cl.run_with_event(queue, kernel, 1024)
      LibCL.cl_wait_for_events(1, pointerof(event))

      start_time = Cl.get_profiling_info(event, LibCL::ClProfilingInfo::PROFILING_COMMAND_START)
      end_time = Cl.get_profiling_info(event, LibCL::ClProfilingInfo::PROFILING_COMMAND_END)

      (end_time >= start_time).should be_true

      Cl.release_event(event)
      Cl.release_buffer(buf)
      Cl.release_kernel(kernel)
      Cl.release_program(program)
      Cl.release_queue(queue)
    end

    it "can allocate and map SVM if supported" do
      device, context, queue = Cl.single_device_defaults
      if Cl.svm_supported?(device)
        ptr = LibCL.cl_svm_alloc(context, LibCL::ClMemFlags::READ_WRITE.to_u64, 1024_u64, 0_u32)
        ptr.should_not be_nil

        begin
          Cl.map_svm(queue, LibCL::CL_TRUE, LibCL::ClMapFlags::WRITE.to_u64, ptr, 1024_u64)
          # Use it as a normal pointer in Crystal
          (ptr.as(Float32*))[0] = 123.45_f32
          Cl.unmap_svm(queue, ptr)
        ensure
          LibCL.cl_svm_free(context, ptr)
        end
      else
        pending! "Device does not support SVM"
      end
    end

    it "can query detailed device information" do
      device, context, queue = Cl.single_device_defaults

      Cl.device_name(device).should_not be_empty
      Cl.global_memory(device).should be > 0
      Cl.local_memory(device).should be > 0
      Cl.max_work_groups(device).should be > 0

      items = Cl.max_work_items(device)
      items.should_not be_empty
      items.each { |dim| dim.should be > 0 }
    end

    it "can perform basic read/write operations on buffers" do
      device, context, queue = Cl.single_device_defaults

      data = [1.0, 2.0, 3.0, 4.0]
      buf = Cl.buffer_like(context, data)

      begin
        Cl.write(queue, data, buf)

        output = Array(Float64).new(4, 0.0)
        Cl.read(queue, output, buf)

        output.should eq(data)
      ensure
        Cl.release_buffer(buf)
      end
    end

    it "can run 2D and 3D kernels" do
      device, context, queue = Cl.single_device_defaults

      program = Cl.create_and_build(context, "
        __kernel void test2d(__global float* a) {
          int x = get_global_id(0);
          int y = get_global_id(1);
          int width = get_global_size(0);
          a[y * width + x] = (float)(x + y);
        }
        __kernel void test3d(__global float* a) {
          int x = get_global_id(0);
          int y = get_global_id(1);
          int z = get_global_id(2);
          int width = get_global_size(0);
          int height = get_global_size(1);
          a[(z * height + y) * width + x] = (float)(x + y + z);
        }
      ", device)

      begin
        k2d = Cl.create_kernel(program, "test2d")
        buf2d = Cl.buffer(context, 16_u64, dtype: Float32)
        Cl.args(k2d, buf2d)
        Cl.run2d(queue, k2d, {4, 4})

        output2d = Array(Float32).new(16, 0.0_f32)
        Cl.read(queue, output2d, buf2d)
        output2d[5].should eq(1.0 + 1.0) # x=1, y=1

        k3d = Cl.create_kernel(program, "test3d")
        buf3d = Cl.buffer(context, 8_u64, dtype: Float32)
        Cl.args(k3d, buf3d)
        Cl.run3d(queue, k3d, {2, 2, 2})

        output3d = Array(Float32).new(8, 0.0_f32)
        Cl.read(queue, output3d, buf3d)
        output3d[7].should eq(1.0 + 1.0 + 1.0) # x=1, y=1, z=1

        Cl.release_kernel(k2d)
        Cl.release_kernel(k3d)
        Cl.release_buffer(buf2d)
        Cl.release_buffer(buf3d)
      ensure
        Cl.release_program(program)
      end
    end

    it "can set kernel arguments of various types" do
      device, context, queue = Cl.single_device_defaults
      program = Cl.create_and_build(context, "
        __kernel void test_args(__global float* a, int b, float d) {
          a[0] = (float)(b + d);
        }
      ", device)

      begin
        kernel = Cl.create_kernel(program, "test_args")
        buf = Cl.buffer(context, 1_u64, dtype: Float32)

        # Test generic args macro
        Cl.args(kernel, buf, 10_i32, 30.5_f32)

        Cl.run(queue, kernel, 1)

        output = Array(Float32).new(1, 0.0_f32)
        Cl.read(queue, output, buf)
        output[0].should eq(10 + 30.5)

        Cl.release_kernel(kernel)
        Cl.release_buffer(buf)
      ensure
        Cl.release_program(program)
      end
    end

    it "can create pipes if supported" do
      device, context, queue = Cl.single_device_defaults
      if Cl.supports_pipes?(device)
        pipe = Cl.create_pipe(context, LibCL::ClMemFlags::READ_WRITE, 4_u32, 10_u32)
        pipe.should_not be_nil
        Cl.release_buffer(pipe)
      else
        pending! "Device does not support Pipes"
      end
    end
  else
    it "skips OpenCL tests (no platform available)" do
      pending! "No OpenCL platform found on the host system"
    end
  end
end
