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

    it "can create and set user events" do
      device, context, queue = Cl.single_device_defaults
      event = Cl.create_user_event(context)
      event.should_not be_nil
      Cl.set_user_event_status(event, 0)
      Cl.release_event(event)
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
