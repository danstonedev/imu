let device, rotatePipeline, relQuatPipeline;

export async function initWebGPU() {
  if (!('gpu' in navigator)) throw new Error('WebGPU not supported');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  device = await adapter.requestDevice();

  const rotateWGSL = /* wgsl */`
struct QBuf { data: array<vec4<f32>>; };
struct VBuf { data: array<vec4<f32>>; };
@group(0) @binding(0) var<storage, read>  q_in : QBuf;
@group(0) @binding(1) var<storage, read>  v_in : VBuf;
@group(0) @binding(2) var<storage, read_write> v_out : VBuf;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&q_in.data)) { return; }
  let q = q_in.data[i];              // [w,x,y,z]
  let v = v_in.data[i].xyz;          // vec3
  let qv = q.yzw;                    // [x,y,z]
  let t  = 2.0 * cross(qv, v);
  let vp = v + q.x * t + cross(qv, t);
  v_out.data[i] = vec4<f32>(vp, 0.0);
}
`;

  const relQuatWGSL = /* wgsl */`
struct QBuf { data: array<vec4<f32>>; };
@group(0) @binding(0) var<storage, read>  parentQ : QBuf;
@group(0) @binding(1) var<storage, read>  childQ  : QBuf;
@group(0) @binding(2) var<storage, read_write> outQ : QBuf;

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
    a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
    a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
    a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x
  );
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> { return vec4<f32>(q.x, -q.y, -q.z, -q.w); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x; if (i >= arrayLength(&parentQ.data)) { return; }
  let qp = parentQ.data[i]; let qc = childQ.data[i];
  outQ.data[i] = quat_mul(quat_conj(qp), qc);
}
`;

  rotatePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: rotateWGSL }), entryPoint: 'main' } });
  relQuatPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: relQuatWGSL }), entryPoint: 'main' } });
  return device;
}

function makeBuf(ta, usage) {
  const buf = device.createBuffer({ size: ta.byteLength, usage, mappedAtCreation: true });
  const view = new ta.constructor(buf.getMappedRange()); view.set(ta); buf.unmap();
  return buf;
}

async function runKernel(pipeline, bindGroup, N) {
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(N / 256));
  pass.end();
  device.queue.submit([enc.finish()]);
}

export async function rotateVecsGPU(qFloat32_4N, vFloat32_3N) {
  const N = qFloat32_4N.length / 4;
  const v4 = new Float32Array(N * 4);
  for (let i = 0, j = 0; i < vFloat32_3N.length; i += 3, j += 4) {
    v4[j] = vFloat32_3N[i]; v4[j+1] = vFloat32_3N[i+1]; v4[j+2] = vFloat32_3N[i+2]; v4[j+3] = 0.0;
  }
  const qBuf = makeBuf(qFloat32_4N, GPUBufferUsage.STORAGE);
  const vIn  = makeBuf(v4,             GPUBufferUsage.STORAGE);
  const vOut = device.createBuffer({ size: v4.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  const bind = device.createBindGroup({ layout: rotatePipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: qBuf } },
    { binding: 1, resource: { buffer: vIn } },
    { binding: 2, resource: { buffer: vOut } },
  ]});

  await runKernel(rotatePipeline, bind, N);

  const readback = device.createBuffer({ size: v4.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(vOut, 0, readback, 0, v4.byteLength);
  device.queue.submit([enc.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readback.getMappedRange()).slice(); readback.unmap();

  const r = new Float32Array(N * 3);
  for (let i = 0, j = 0; j < out.length; i += 3, j += 4) { r[i]=out[j]; r[i+1]=out[j+1]; r[i+2]=out[j+2]; }
  return r;
}

export async function relativeQuatGPU(qParent_4N, qChild_4N) {
  const N = qParent_4N.length / 4;
  const parentBuf = makeBuf(qParent_4N, GPUBufferUsage.STORAGE);
  const childBuf  = makeBuf(qChild_4N,  GPUBufferUsage.STORAGE);
  const outBuf    = device.createBuffer({ size: qParent_4N.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  const bind = device.createBindGroup({ layout: relQuatPipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: parentBuf } },
    { binding: 1, resource: { buffer: childBuf } },
    { binding: 2, resource: { buffer: outBuf } },
  ]});

  await runKernel(relQuatPipeline, bind, N);

  const readback = device.createBuffer({ size: qParent_4N.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(outBuf, 0, readback, 0, qParent_4N.byteLength);
  device.queue.submit([enc.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readback.getMappedRange()).slice(); readback.unmap();
  return out;
}
