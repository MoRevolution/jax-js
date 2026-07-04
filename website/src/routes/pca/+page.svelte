<script lang="ts">
  import { browser } from "$app/environment";
  import { resolve } from "$app/paths";

  import {
    blockUntilReady,
    defaultDevice,
    init,
    numpy as np,
  } from "@jax-js/jax";
  import { Cpu, RefreshCw, Rotate3d, Zap } from "@lucide/svelte";
  import { onDestroy, onMount, tick } from "svelte";
  import * as THREE from "three";
  import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

  type DeviceChoice = "wasm" | "webgpu";

  type Point3 = {
    x: number;
    y: number;
    z: number;
    tone: number;
  };

  type Vec3 = [number, number, number];

  type PcaResult = {
    mean: Vec3;
    values: number[];
    components: Vec3[];
    scores: [number, number][];
    elapsedMs: number;
    device: DeviceChoice;
  };

  let sceneHost: HTMLDivElement;
  let renderer: THREE.WebGLRenderer | null = null;
  let scene: THREE.Scene | null = null;
  let camera: THREE.PerspectiveCamera | null = null;
  let controls: OrbitControls | null = null;
  let resizeObserver: ResizeObserver | null = null;
  let frame = 0;
  let worldGroup: THREE.Group | null = null;
  let pointsObject: THREE.Points | null = null;
  let axesGroup: THREE.Group | null = null;

  let points = $state<Point3[]>([]);
  let pca = $state<PcaResult | null>(null);
  let availableDevices = $state<DeviceChoice[]>(["wasm"]);
  let device = $state<DeviceChoice>("wasm");
  let pointCount = $state(1200);
  let noise = $state(0.42);
  let seed = $state(0);
  let running = $state(false);
  let error = $state<string | null>(null);
  let autoRotate = $state(true);

  const pcColors = [0xc43b20, 0x00796f, 0x5642a7];
  const cameraDirection = new THREE.Vector3(7.4, 5.2, 8.2).normalize();

  function mulberry32(a: number) {
    return () => {
      let t = (a += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function normal(rand: () => number) {
    const u = Math.max(rand(), 1e-8);
    const v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function randomSeed() {
    const value = new Uint32Array(1);
    crypto.getRandomValues(value);
    return value[0];
  }

  function normalize3(vector: Vec3): Vec3 {
    const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
    return [vector[0] / length, vector[1] / length, vector[2] / length];
  }

  function cross3(a: Vec3, b: Vec3): Vec3 {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }

  function randomBasis(rand: () => number): [Vec3, Vec3, Vec3] {
    const first = normalize3([normal(rand), normal(rand), normal(rand)]);
    const rawSecond = normalize3([normal(rand), normal(rand), normal(rand)]);
    const secondProjection = dot3(rawSecond, first);
    const second = normalize3([
      rawSecond[0] - first[0] * secondProjection,
      rawSecond[1] - first[1] * secondProjection,
      rawSecond[2] - first[2] * secondProjection,
    ]);
    return [first, second, cross3(first, second)];
  }

  function combineBasis(basis: [Vec3, Vec3, Vec3], values: Vec3): Vec3 {
    return [
      basis[0][0] * values[0] +
        basis[1][0] * values[1] +
        basis[2][0] * values[2],
      basis[0][1] * values[0] +
        basis[1][1] * values[1] +
        basis[2][1] * values[2],
      basis[0][2] * values[0] +
        basis[1][2] * values[1] +
        basis[2][2] * values[2],
    ];
  }

  function generatePoints(
    count: number,
    noiseScale: number,
    seedValue: number,
  ) {
    const rand = mulberry32(seedValue);
    const basis = randomBasis(rand);
    const center: Vec3 = [
      (rand() - 0.5) * 0.7,
      (rand() - 0.5) * 0.7,
      (rand() - 0.5) * 0.7,
    ];
    const primaryScale = 2.15 + rand() * 0.85;
    const secondaryScale = 0.78 + rand() * 0.5;
    const tertiaryScale = 0.64 + rand() * 0.34;
    const wavePhase = rand() * Math.PI * 2;
    const generated: Point3[] = [];
    for (let i = 0; i < count; i++) {
      const cluster = i % 3;
      const offset = cluster === 0 ? -1.15 : cluster === 1 ? 0.15 : 1.05;
      const primary = normal(rand) * 1.05 + offset;
      const u = primary * primaryScale;
      const v =
        normal(rand) * secondaryScale + 0.28 * Math.sin(i * 0.011 + wavePhase);
      const w = normal(rand) * noiseScale * tertiaryScale;
      const [x, y, z] = combineBasis(basis, [u, v, w]);
      generated.push({
        x: x + center[0],
        y: y + center[1],
        z: z + center[2],
        tone: Math.max(0, Math.min(1, (primary + 3.2) / 6.4)),
      });
    }
    return generated;
  }

  function orientComponent(component: Vec3): Vec3 {
    const sign =
      component[0] + component[1] * 0.4 - component[2] * 0.2 < 0 ? -1 : 1;
    return [component[0] * sign, component[1] * sign, component[2] * sign];
  }

  async function computePca() {
    if (!browser || running) return;
    running = true;
    error = null;

    try {
      const devices = (await init("wasm", "webgpu")) as DeviceChoice[];
      availableDevices = devices.filter(
        (value): value is DeviceChoice =>
          value === "wasm" || value === "webgpu",
      );
      if (!availableDevices.includes(device)) {
        device = availableDevices.includes("webgpu") ? "webgpu" : "wasm";
      }
      defaultDevice(device);

      const cloud = generatePoints(pointCount, noise, seed);
      points = cloud;
      await tick();
      updateScene();

      const data = new Float32Array(cloud.length * 3);
      for (let i = 0; i < cloud.length; i++) {
        data[i * 3] = cloud[i].x;
        data[i * 3 + 1] = cloud[i].y;
        data[i * 3 + 2] = cloud[i].z;
      }

      const start = performance.now();
      const x = np.array(data, { shape: [cloud.length, 3], device });
      const mean = np.mean(x.ref, 0);
      const centered = x.ref.sub(mean.ref.reshape([1, 3]));
      const covariance = np
        .matmul(np.matrixTranspose(centered.ref), centered.ref)
        .div(cloud.length - 1);
      const [valuesAsc, vectorsAsc] = np.linalg.eigh(covariance, {
        symmetrizeInput: false,
      });
      await blockUntilReady([valuesAsc, vectorsAsc]);
      const elapsedMs = performance.now() - start;

      const valuesRaw = (await valuesAsc.ref.jsAsync()) as number[];
      const vectorsRaw = (await vectorsAsc.ref.jsAsync()) as number[][];
      const meanRaw = (await mean.ref.jsAsync()) as Vec3;
      const order = [2, 1, 0];
      const values = order.map((index) => Math.max(valuesRaw[index], 0));
      const components = order.map((index) =>
        orientComponent([
          vectorsRaw[0][index],
          vectorsRaw[1][index],
          vectorsRaw[2][index],
        ]),
      );

      const scores = cloud.map((point) => {
        const centeredPoint: Vec3 = [
          point.x - meanRaw[0],
          point.y - meanRaw[1],
          point.z - meanRaw[2],
        ];
        return [
          dot3(centeredPoint, components[0]),
          dot3(centeredPoint, components[1]),
        ] as [number, number];
      });

      pca = {
        mean: meanRaw,
        values,
        components,
        scores,
        elapsedMs,
        device,
      };

      x.dispose();
      mean.dispose();
      centered.dispose();
      valuesAsc.dispose();
      vectorsAsc.dispose();

      await tick();
      updateScene();
    } catch (cause) {
      console.error(cause);
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      running = false;
    }
  }

  function dot3(a: Vec3, b: Vec3) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function initScene() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf7f7f0);

    camera = new THREE.PerspectiveCamera(42, 1, 0.1, 120);
    camera.zoom = 2;
    camera.position.set(7.4, 5.2, 8.2);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    sceneHost.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);

    const ambient = new THREE.AmbientLight(0xffffff, 1.8);
    scene.add(ambient);
    const light = new THREE.DirectionalLight(0xffffff, 1.6);
    light.position.set(5, 7, 3);
    scene.add(light);

    const grid = new THREE.GridHelper(10, 10, 0xd7d2c1, 0xe8e3d4);
    grid.position.y = -3.2;
    scene.add(grid);

    worldGroup = new THREE.Group();
    scene.add(worldGroup);

    resizeObserver = new ResizeObserver(resizeScene);
    resizeObserver.observe(sceneHost);
    resizeScene();
    animate();
  }

  function resizeScene() {
    if (!sceneHost || !renderer || !camera) return;
    const width = Math.max(1, sceneHost.clientWidth);
    const height = Math.max(1, sceneHost.clientHeight);
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    frameScene();
  }

  function animate() {
    if (!renderer || !scene || !camera) return;
    if (worldGroup && autoRotate) worldGroup.rotation.y += 0.0025;
    controls?.update();
    renderer.render(scene, camera);
    frame = requestAnimationFrame(animate);
  }

  function pointColor(tone: number) {
    const color = new THREE.Color();
    const value = Math.max(0, Math.min(1, tone));
    if (value < 0.5) {
      color.lerpColors(
        new THREE.Color(0x37cfe0),
        new THREE.Color(0x5ed75f),
        value * 2,
      );
    } else {
      color.lerpColors(
        new THREE.Color(0x5ed75f),
        new THREE.Color(0xefb53b),
        (value - 0.5) * 2,
      );
    }
    return color;
  }

  function createPointMaterial() {
    return new THREE.ShaderMaterial({
      vertexColors: true,
      transparent: true,
      uniforms: {
        pointSize: { value: 4.6 * Math.min(window.devicePixelRatio, 2) },
      },
      vertexShader: `
        uniform float pointSize;
        varying vec3 vColor;

        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mvPosition;
          gl_PointSize = pointSize;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;

        void main() {
          vec2 centerOffset = gl_PointCoord - vec2(0.5);
          float distanceFromCenter = length(centerOffset);
          float alpha = 1.0 - smoothstep(0.42, 0.5, distanceFromCenter);
          if (alpha <= 0.0) discard;
          gl_FragColor = vec4(vColor, alpha);
        }
      `,
    });
  }

  function createArrowMaterial(color: number) {
    return new THREE.MeshBasicMaterial({
      color,
      depthTest: false,
      depthWrite: false,
    });
  }

  function createComponentArrow(
    direction: THREE.Vector3,
    origin: THREE.Vector3,
    length: number,
    color: number,
    boundsRadius: number,
    headScale = 1,
  ) {
    const group = new THREE.Group();
    const unit = direction.clone().normalize();
    const rotation = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      unit,
    );
    const headLength =
      Math.max(0.09, Math.min(0.21, boundsRadius * 0.0325)) * headScale;
    const shaftLength = Math.max(0.01, length - headLength);
    const shaftRadius = Math.max(
      0.009,
      Math.min(0.0225, boundsRadius * 0.0035),
    );
    const headRadius = headLength * 0.38;

    const shaft = new THREE.Mesh(
      new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLength, 16),
      createArrowMaterial(color),
    );
    shaft.quaternion.copy(rotation);
    shaft.position.copy(origin).addScaledVector(unit, shaftLength * 0.5);
    shaft.renderOrder = 10;

    const head = new THREE.Mesh(
      new THREE.ConeGeometry(headRadius, headLength, 20),
      createArrowMaterial(color),
    );
    head.quaternion.copy(rotation);
    head.position
      .copy(origin)
      .addScaledVector(unit, shaftLength + headLength * 0.5);
    head.renderOrder = 10;

    group.add(shaft, head);
    return group;
  }

  function updateScene() {
    if (!worldGroup) return;

    if (pointsObject) {
      worldGroup.remove(pointsObject);
      pointsObject.geometry.dispose();
      (pointsObject.material as THREE.Material).dispose();
      pointsObject = null;
    }
    if (axesGroup) {
      worldGroup.remove(axesGroup);
      axesGroup.traverse((object) => {
        const mesh = object as THREE.Mesh;
        mesh.geometry?.dispose?.();
        const material = mesh.material as THREE.Material | undefined;
        material?.dispose?.();
      });
      axesGroup = null;
    }

    const bounds = measurePointBounds();

    if (points.length) {
      const positions = new Float32Array(points.length * 3);
      const colors = new Float32Array(points.length * 3);
      for (let i = 0; i < points.length; i++) {
        const point = points[i];
        positions[i * 3] = point.x;
        positions[i * 3 + 1] = point.y;
        positions[i * 3 + 2] = point.z;
        const color = pointColor(point.tone);
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3),
      );
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      const material = createPointMaterial();
      pointsObject = new THREE.Points(geometry, material);
      worldGroup.add(pointsObject);
    }

    if (pca) {
      axesGroup = new THREE.Group();
      const origin = new THREE.Vector3(...pca.mean);
      for (let i = 0; i < pca.components.length; i++) {
        const direction = new THREE.Vector3(...pca.components[i]).normalize();
        const rawLength = Math.sqrt(pca.values[i]) * (i === 0 ? 2.05 : 1.65);
        const length = Math.min(
          rawLength,
          bounds.radius * (i === 0 ? 0.78 : 0.58),
        );
        const forward = createComponentArrow(
          direction,
          origin,
          length,
          pcColors[i],
          bounds.radius,
        );
        const backward = createComponentArrow(
          direction.clone().multiplyScalar(-1),
          origin,
          length,
          pcColors[i],
          bounds.radius,
          0.8,
        );
        axesGroup.add(forward, backward);
      }
      worldGroup.add(axesGroup);
    }

    frameScene(bounds);
  }

  function measurePointBounds() {
    if (!points.length) {
      return {
        center: new THREE.Vector3(0, 0, 0),
        radius: 5,
      };
    }

    const box = new THREE.Box3();
    const pointVector = new THREE.Vector3();
    for (const point of points) {
      box.expandByPoint(pointVector.set(point.x, point.y, point.z));
    }

    const center = box.getCenter(new THREE.Vector3());
    let radius = box.getSize(new THREE.Vector3()).length() * 0.5;
    for (const point of points) {
      radius = Math.max(
        radius,
        center.distanceTo(pointVector.set(point.x, point.y, point.z)),
      );
    }

    return {
      center,
      radius: Math.max(1, radius),
    };
  }

  function frameScene(bounds = measurePointBounds()) {
    if (!camera || !controls || !points.length) return;

    const verticalFov = THREE.MathUtils.degToRad(camera.fov);
    const horizontalFov =
      2 * Math.atan(Math.tan(verticalFov * 0.5) * camera.aspect);
    const fitFov = Math.max(0.001, Math.min(verticalFov, horizontalFov));
    const distance = (bounds.radius * 0.72) / Math.sin(fitFov * 0.5);

    camera.position
      .copy(bounds.center)
      .addScaledVector(cameraDirection, distance);
    camera.near = Math.max(0.01, distance - bounds.radius * 4);
    camera.far = distance + bounds.radius * 4 + 50;
    camera.updateProjectionMatrix();

    controls.target.copy(bounds.center);
    controls.minDistance = Math.max(0.5, bounds.radius * 0.35);
    controls.maxDistance = Math.max(20, bounds.radius * 8);
    controls.update();
  }

  function projectionPoint(score: [number, number]) {
    if (!pca) return { x: 0, y: 0 };
    const maxX = Math.max(1e-6, Math.sqrt(pca.values[0]) * 3.2);
    const maxY = Math.max(1e-6, Math.sqrt(pca.values[1]) * 3.2);
    return {
      x: 50 + (score[0] / maxX) * 42,
      y: 50 - (score[1] / maxY) * 42,
    };
  }

  function explainedVariance(index: number) {
    if (!pca) return 0;
    const total = pca.values.reduce((sum, value) => sum + value, 0);
    return total > 0 ? pca.values[index] / total : 0;
  }

  onMount(async () => {
    seed = randomSeed();
    initScene();
    await computePca();
  });

  onDestroy(() => {
    if (!browser) return;
    cancelAnimationFrame(frame);
    resizeObserver?.disconnect();
    controls?.dispose();
    if (renderer) {
      renderer.dispose();
      renderer.domElement.remove();
    }
  });
</script>

<svelte:head>
  <title>PCA demo - jax-js</title>
</svelte:head>

<main class="min-h-screen bg-[#f7f7f0] font-tiktok text-[#202019]">
  <header
    class="mx-auto flex max-w-screen-2xl items-center justify-between gap-4 px-4 py-4 sm:px-6"
  >
    <a href={resolve("/")} class="text-sm font-medium text-[#202019]">jax-js</a>
    <div class="flex items-center gap-2 text-xs text-[#6d6b5f]">
      <span class="hidden sm:inline">PCA</span>
      <span class="h-1.5 w-1.5 rounded-full bg-[#96a026]"></span>
      <span
        >{pca
          ? `${pca.elapsedMs.toFixed(1)} ms`
          : running
            ? "running"
            : "ready"}</span
      >
    </div>
  </header>

  <section class="border-y border-[#202019]/15">
    <div
      class="mx-auto grid max-w-screen-2xl lg:grid-cols-[minmax(0,1fr)_25rem]"
    >
      <div class="h-[64vh] min-h-[30rem] lg:h-[calc(100vh-4rem)]">
        <div bind:this={sceneHost} class="h-full w-full"></div>
      </div>

      <aside
        class="border-t border-[#202019]/15 bg-[#fbfaf5] lg:border-l lg:border-t-0"
      >
        <div class="space-y-6 p-4 sm:p-6">
          <div>
            <p
              class="font-mono text-xs uppercase tracking-[0.18em] text-[#777365]"
            >
              Principal components
            </p>
            <h1 class="mt-2 text-3xl font-semibold leading-tight">PCA demo</h1>
            <p class="mt-3 text-sm leading-relaxed text-[#686658]">
              3D point cloud, covariance eigendecomposition, and the PC1/PC2
              projection.
            </p>
          </div>

          <div class="grid grid-cols-2 gap-3 text-sm">
            <label class="space-y-2">
              <span class="text-xs text-[#686658]">Points</span>
              <input
                type="range"
                min="240"
                max="2400"
                step="120"
                bind:value={pointCount}
                class="w-full accent-[#96a026]"
              />
              <span class="font-mono text-xs">{pointCount}</span>
            </label>

            <label class="space-y-2">
              <span class="text-xs text-[#686658]">Noise</span>
              <input
                type="range"
                min="0.05"
                max="1.2"
                step="0.05"
                bind:value={noise}
                class="w-full accent-[#96a026]"
              />
              <span class="font-mono text-xs">{noise.toFixed(2)}</span>
            </label>
          </div>

          <div class="grid grid-cols-[1fr_auto_auto] gap-2">
            <label class="relative">
              <span class="sr-only">Backend</span>
              <select
                bind:value={device}
                disabled={running}
                class="h-10 w-full border border-[#202019]/20 bg-white px-3 text-sm outline-none focus:border-[#96a026]"
              >
                {#each availableDevices as option}
                  <option value={option}>{option}</option>
                {/each}
              </select>
            </label>
            <button
              class="inline-flex h-10 w-10 items-center justify-center border border-[#202019]/20 bg-white transition hover:bg-[#f0f1e6] disabled:opacity-50"
              disabled={running}
              title="Regenerate"
              onclick={() => {
                seed += 1;
                computePca();
              }}
            >
              <RefreshCw size={17} class={running ? "animate-spin" : ""} />
            </button>
            <button
              class="inline-flex h-10 w-10 items-center justify-center border border-[#202019]/20 bg-white transition hover:bg-[#f0f1e6]"
              class:bg-[#e8ead4]={autoRotate}
              title="Rotate"
              onclick={() => (autoRotate = !autoRotate)}
            >
              <Rotate3d size={17} />
            </button>
          </div>

          {#if error}
            <p
              class="border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-900"
            >
              {error}
            </p>
          {/if}

          <section class="border border-[#202019]/15 bg-white">
            <div
              class="border-b border-[#202019]/15 px-3 py-2 text-xs text-[#686658]"
            >
              PC1 x PC2
            </div>
            <svg
              viewBox="0 0 100 100"
              class="aspect-square w-full bg-[#f9f8f1]"
            >
              <line
                x1="8"
                y1="50"
                x2="92"
                y2="50"
                stroke="#d7d2c1"
                stroke-width="0.35"
              />
              <line
                x1="50"
                y1="8"
                x2="50"
                y2="92"
                stroke="#d7d2c1"
                stroke-width="0.35"
              />
              {#if pca}
                {#each pca.scores as score, i}
                  {@const projected = projectionPoint(score)}
                  <circle
                    cx={projected.x}
                    cy={projected.y}
                    r="0.55"
                    fill={`hsl(${188 - points[i].tone * 150} 68% 45%)`}
                    opacity="0.72"
                  />
                {/each}
              {/if}
            </svg>
          </section>

          <section class="grid gap-2">
            {#each [0, 1, 2] as index}
              <div
                class="grid grid-cols-[3rem_1fr_4rem] items-center gap-3 text-sm"
              >
                <span
                  class="font-mono text-xs"
                  style={`color: #${pcColors[index].toString(16)}`}
                >
                  PC{index + 1}
                </span>
                <div class="h-2 bg-[#e3dfd0]">
                  <div
                    class="h-full"
                    style={`width: ${Math.max(2, explainedVariance(index) * 100)}%; background: #${pcColors[index].toString(16)}`}
                  ></div>
                </div>
                <span class="text-right font-mono text-xs">
                  {(explainedVariance(index) * 100).toFixed(1)}%
                </span>
              </div>
            {/each}
          </section>

          <div class="grid grid-cols-2 gap-2 text-xs text-[#686658]">
            <div
              class="flex items-center gap-2 border border-[#202019]/15 bg-white px-3 py-2"
            >
              <Cpu size={15} />
              <span>{pca?.device ?? device}</span>
            </div>
            <div
              class="flex items-center gap-2 border border-[#202019]/15 bg-white px-3 py-2"
            >
              <Zap size={15} />
              <span>{pca ? `${pca.elapsedMs.toFixed(1)} ms` : "-"}</span>
            </div>
          </div>
        </div>
      </aside>
    </div>
  </section>
</main>
