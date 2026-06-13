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
  else
    it "skips OpenCL tests (no platform available)" do
      pending! "No OpenCL platform found on the host system"
    end
  end
end
