import { useRef, useEffect, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Book, ColourMode, HoveredPassage, ViewMode, Metadata, InterpolateState, Passage } from '../utils/types';
import { BOOK_COLOURS, getFeatureColour, magnitudeColour } from '../utils/colours';

interface Props {
  books: Book[];
  metadata: Metadata | null;
  activeBooks: Set<number>;
  colourMode: ColourMode;
  viewMode: ViewMode;
  fieldEnabled: boolean;
  interpolate: InterpolateState;
  hovered: HoveredPassage | null;
  onHover: (h: HoveredPassage | null) => void;
  playback: { bookIndex: number; passageIndex: number } | null;
  hoveredBookIndex: number | null;
}

function hexToThreeColor(hex: string): THREE.Color {
  return new THREE.Color(hex);
}

/** Get the 3D coordinate of a passage in the current view mode. Falls
 * back to default coords if canonical is unavailable. */
function passageCoord(p: Passage, view: ViewMode): [number, number, number] {
  if (view === 'canonical' && p.canonical) {
    return [p.canonical[0], p.canonical[1], p.canonical[2]];
  }
  return [p.x3d, p.y3d, p.z3d];
}

function centerOfData(books: Book[], view: ViewMode) {
  let sx = 0, sy = 0, sz = 0, n = 0;
  for (const b of books) {
    for (const p of b.passages) {
      const [x, y, z] = passageCoord(p, view);
      sx += x; sy += y; sz += z; n++;
    }
  }
  if (n === 0) return new THREE.Vector3(0, 0, 0);
  return new THREE.Vector3(sx / n, sy / n, sz / n);
}

export function Canvas({ books, metadata, activeBooks, colourMode, viewMode, fieldEnabled, interpolate, hovered, onHover, playback, hoveredBookIndex }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    raycaster: THREE.Raycaster;
    mouse: THREE.Vector2;
    pointsGroup: THREE.Group;
    linesGroup: THREE.Group;
    highlightGroup: THREE.Group;
    fieldGroup: THREE.Group;
    interpGroup: THREE.Group;
    // For hit-testing: flat arrays
    allPositions: { bookIndex: number; passageIndex: number; position: THREE.Vector3 }[];
    pointsMesh: THREE.Points | null;
  } | null>(null);
  const booksRef = useRef(books);
  booksRef.current = books;

  // Init Three.js scene
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#f5f0e8');

    const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 200);
    camera.position.set(20, 15, 20);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 3;
    controls.maxDistance = 60;

    // Subtle ambient + directional light
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);

    const pointsGroup = new THREE.Group();
    const linesGroup = new THREE.Group();
    const highlightGroup = new THREE.Group();
    const fieldGroup = new THREE.Group();
    const interpGroup = new THREE.Group();
    scene.add(pointsGroup);
    scene.add(linesGroup);
    scene.add(highlightGroup);
    scene.add(fieldGroup);
    scene.add(interpGroup);

    stateRef.current = {
      scene, camera, renderer, controls,
      raycaster: new THREE.Raycaster(),
      mouse: new THREE.Vector2(),
      pointsGroup, linesGroup, highlightGroup, fieldGroup, interpGroup,
      allPositions: [],
      pointsMesh: null,
    };

    // Animation loop
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    });
    ro.observe(container);

    return () => {
      cancelAnimationFrame(animId);
      ro.disconnect();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  // Center camera on data once loaded, and again when view mode changes
  // (canonical coords live in a different range from default).
  useEffect(() => {
    if (!stateRef.current || books.length === 0) return;
    const center = centerOfData(books, viewMode);
    stateRef.current.controls.target.copy(center);
    stateRef.current.camera.position.set(center.x + 12, center.y + 8, center.z + 12);
    stateRef.current.controls.update();
  }, [books.length === 0, viewMode]); // eslint-disable-line react-hooks/exhaustive-deps

  // Rebuild points and lines whenever relevant state changes
  useEffect(() => {
    const s = stateRef.current;
    if (!s || books.length === 0) return;

    // Clear previous
    s.pointsGroup.clear();
    s.linesGroup.clear();
    s.highlightGroup.clear();
    s.allPositions = [];

    // Build all positions + colours
    const positions: number[] = [];
    const colors: number[] = [];
    const sizes: number[] = [];

    for (let bi = 0; bi < books.length; bi++) {
      const book = books[bi];
      const bookColour = hexToThreeColor(BOOK_COLOURS[bi % BOOK_COLOURS.length]);
      const isActive = activeBooks.has(bi);

      const isHighlighted = hoveredBookIndex === bi;
      const isDimmed = hoveredBookIndex !== null && hoveredBookIndex !== bi;

      for (let pi = 0; pi < book.passages.length; pi++) {
        const p = book.passages[pi];
        const [px, py, pz] = passageCoord(p, viewMode);
        const pos = new THREE.Vector3(px, py, pz);
        positions.push(pos.x, pos.y, pos.z);

        s.allPositions.push({ bookIndex: bi, passageIndex: pi, position: pos });

        let color: THREE.Color;
        if (colourMode === 'book') {
          if (isActive || isHighlighted) {
            color = bookColour.clone();
          } else {
            color = bookColour.clone().lerp(new THREE.Color('#d8d0c2'), 0.4);
          }
        } else {
          const fc = getFeatureColour(colourMode, p, book);
          color = hexToThreeColor(fc);
          if (!isActive && !isHighlighted) color.lerp(new THREE.Color('#d8d0c2'), 0.3);
        }
        // Dim: hover in sidebar takes priority, then selection-based dimming
        const hasSelection = activeBooks.size > 0;
        if (isDimmed) {
          color.lerp(new THREE.Color('#e8e0d4'), 0.75);
        } else if (hasSelection && !isActive && !isHighlighted) {
          color.lerp(new THREE.Color('#e8e0d4'), 0.6);
        }

        colors.push(color.r, color.g, color.b);
        const size = isActive || isHighlighted ? 0.22
          : isDimmed ? 0.08
          : hasSelection ? 0.08
          : 0.13;
        sizes.push(size);
      }
    }

    // Points (instanced spheres for quality)
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geom.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

    const pointsMaterial = new THREE.ShaderMaterial({
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      uniforms: {
        pixelRatio: { value: window.devicePixelRatio },
      },
      vertexShader: `
        attribute float size;
        varying vec3 vColor;
        uniform float pixelRatio;
        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * pixelRatio * (300.0 / -mvPosition.z);
          gl_PointSize = max(gl_PointSize, 2.5);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        void main() {
          float d = length(gl_PointCoord - vec2(0.5));
          if (d > 0.5) discard;
          float alpha = 1.0 - smoothstep(0.35, 0.5, d);
          gl_FragColor = vec4(vColor, alpha * 0.9);
        }
      `,
    });

    const pointsMesh = new THREE.Points(geom, pointsMaterial);
    s.pointsGroup.add(pointsMesh);
    s.pointsMesh = pointsMesh;

    // Lines for active books
    for (const bi of activeBooks) {
      const book = books[bi];
      const bookColour = hexToThreeColor(BOOK_COLOURS[bi % BOOK_COLOURS.length]);
      const linePositions: number[] = [];

      for (const p of book.passages) {
        const [x, y, z] = passageCoord(p, viewMode);
        linePositions.push(x, y, z);
      }

      const lineGeom = new THREE.BufferGeometry();
      lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
      const lineMat = new THREE.LineBasicMaterial({
        color: bookColour,
        transparent: true,
        opacity: 0.35,
        linewidth: 1,
      });
      s.linesGroup.add(new THREE.Line(lineGeom, lineMat));

      // Start marker — small sphere
      if (book.passages.length > 0) {
        const p0 = book.passages[0];
        const [x0, y0, z0] = passageCoord(p0, viewMode);
        const startGeom = new THREE.SphereGeometry(0.15, 12, 12);
        const startMat = new THREE.MeshBasicMaterial({ color: bookColour });
        const startMesh = new THREE.Mesh(startGeom, startMat);
        startMesh.position.set(x0, y0, z0);
        s.linesGroup.add(startMesh);

        // White ring
        const ringGeom = new THREE.RingGeometry(0.17, 0.22, 16);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x3d3328, side: THREE.DoubleSide });
        const ring = new THREE.Mesh(ringGeom, ringMat);
        ring.position.copy(startMesh.position);
        ring.lookAt(s.camera.position);
        s.linesGroup.add(ring);
      }

      // End marker — small box
      if (book.passages.length > 1) {
        const pN = book.passages[book.passages.length - 1];
        const [xn, yn, zn] = passageCoord(pN, viewMode);
        const endGeom = new THREE.BoxGeometry(0.2, 0.2, 0.2);
        const endMat = new THREE.MeshBasicMaterial({ color: bookColour });
        const endMesh = new THREE.Mesh(endGeom, endMat);
        endMesh.position.set(xn, yn, zn);
        s.linesGroup.add(endMesh);
      }
    }
  }, [books, activeBooks, colourMode, hoveredBookIndex, viewMode]);

  // Playback highlight
  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;
    s.highlightGroup.clear();

    if (playback) {
      const book = books[playback.bookIndex];
      const bookColour = hexToThreeColor(BOOK_COLOURS[playback.bookIndex % BOOK_COLOURS.length]);
      const trailStart = Math.max(0, playback.passageIndex - 10);

      for (let pi = trailStart; pi <= playback.passageIndex; pi++) {
        const p = book.passages[pi];
        const [x, y, z] = passageCoord(p, viewMode);
        const age = playback.passageIndex - pi;
        const t = 1 - age / 11;
        const radius = 0.08 + t * 0.12;
        const geom = new THREE.SphereGeometry(radius, 10, 10);
        const mat = new THREE.MeshBasicMaterial({
          color: bookColour,
          transparent: true,
          opacity: t * 0.9,
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.position.set(x, y, z);
        s.highlightGroup.add(mesh);
      }

      // Glow ring on current
      const curr = book.passages[playback.passageIndex];
      const [cx, cy, cz] = passageCoord(curr, viewMode);
      const ringGeom = new THREE.RingGeometry(0.25, 0.32, 24);
      const ringMat = new THREE.MeshBasicMaterial({
        color: bookColour,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(ringGeom, ringMat);
      ring.position.set(cx, cy, cz);
      ring.lookAt(s.camera.position);
      s.highlightGroup.add(ring);
    }

    if (hovered) {
      const p = books[hovered.bookIndex].passages[hovered.passageIndex];
      const [hx, hy, hz] = passageCoord(p, viewMode);
      const bookColour = hexToThreeColor(BOOK_COLOURS[hovered.bookIndex % BOOK_COLOURS.length]);
      const ringGeom = new THREE.RingGeometry(0.2, 0.26, 24);
      const ringMat = new THREE.MeshBasicMaterial({
        color: bookColour,
        side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(ringGeom, ringMat);
      ring.position.set(hx, hy, hz);
      ring.lookAt(s.camera.position);
      s.highlightGroup.add(ring);
    }
  }, [books, playback, hovered, viewMode]);

  // E04 — vector field overlay.
  //
  // Each grid node is rendered as a short line segment along v(z), with:
  //  - colour mapped to ||v|| via the perceptually graded magnitudeColour
  //    palette (deep navy = slow, vermilion = fast),
  //  - per-vertex colour gradient from a dimmed tail to a fully saturated
  //    head: this *is* the direction indicator, no extra geometry needed,
  //  - per-arrow length scaled by the cell's half-size `h` (adaptive
  //    octree mode) so densely-refined regions read as small clusters
  //    of fine arrows and smooth regions as a few large ones.
  //
  // Backward-compat: if the field has no `h` field per node (uniform
  // grid produced by an older e04 run), fall back to a derived step
  // and apply a magnitude floor to drop the dead zones.
  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;
    s.fieldGroup.clear();
    if (!fieldEnabled || !metadata?.field) return;
    if (viewMode !== 'default') return;

    const grid = metadata.field.grid;
    if (grid.length === 0) return;

    const isAdaptive = grid[0].h !== undefined;

    const norms = grid.map((g) => Math.hypot(g.vx, g.vy, g.vz));
    const sortedNorms = [...norms].sort((a, b) => a - b);
    const maxNorm = sortedNorms[sortedNorms.length - 1] || 1;
    const medNorm = sortedNorms[Math.floor(sortedNorms.length / 2)] || 1;

    // Uniform-grid fallback: cull bottom 30% by magnitude.
    const cullCutoff = isAdaptive ? -1 : sortedNorms[Math.floor(sortedNorms.length * 0.30)];

    // Median half-size — used to normalise per-cell length scales so the
    // visual length doesn't blow up at the largest cells.
    const sortedH = isAdaptive
      ? [...grid.map((g) => g.h ?? 0)].sort((a, b) => a - b)
      : [];
    const medH = isAdaptive ? sortedH[Math.floor(sortedH.length / 2)] || 1 : 1;

    // Uniform grid: derive step from grid_n.
    const gn = metadata.field.grid_n || Math.round(Math.cbrt(grid.length));
    let xmin = Infinity, xmax = -Infinity;
    for (const g of grid) { if (g.x < xmin) xmin = g.x; if (g.x > xmax) xmax = g.x; }
    const uniformStep = (xmax - xmin) / Math.max(1, gn - 1);
    const uniformArrowScale = (uniformStep * 0.9) / medNorm;

    const positions: number[] = [];
    const colours: number[] = [];

    for (let i = 0; i < grid.length; i++) {
      if (!isAdaptive && norms[i] < cullCutoff) continue;
      const g = grid[i];

      let arrowScale: number;
      if (isAdaptive) {
        // Per-arrow scale from cell size. sqrt() compresses the size
        // range so the smallest cells (densest regions) are not
        // invisible: cells 1/8 the median end up ~0.35× the length.
        const h = g.h ?? medH;
        const sizeFactor = Math.sqrt(h / medH);
        // Use the cell's diagonal-ish scale; 1.6× h means the arrow
        // reaches roughly to the cell boundary at the median magnitude.
        arrowScale = (1.6 * medH * sizeFactor) / Math.max(medNorm, norms[i]);
      } else {
        arrowScale = uniformArrowScale;
      }

      const lx = g.x + g.vx * arrowScale;
      const ly = g.y + g.vy * arrowScale;
      const lz = g.z + g.vz * arrowScale;

      const t = norms[i] / maxNorm;
      const [r, gr, b] = magnitudeColour(t);

      // Tail at the cell centre (dim, desaturated), head at the arrow
      // tip (full magnitude colour). The colour gradient itself is the
      // direction indicator.
      const tailFade = 0.18;
      positions.push(g.x, g.y, g.z, lx, ly, lz);
      colours.push(r * tailFade, gr * tailFade, b * tailFade, r, gr, b);
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
    const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 });
    s.fieldGroup.add(new THREE.LineSegments(geom, mat));
  }, [metadata, fieldEnabled, viewMode]);

  // E06 — interpolation polyline. When in 'interpolate' view and both
  // books are picked, draw the lerp between matched-by-position passages
  // of A and B. If `interpolate.fieldInfluence > 0` and an E04 vector
  // field is loaded, each interpolated point is additionally bent along
  // v(z) by `fieldInfluence` × a fixed scale, so the trajectory becomes
  // sensitive to the dynamics of the corpus's drift field.
  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;
    s.interpGroup.clear();
    if (viewMode !== 'interpolate') return;
    const { bookA, bookB, alpha, fieldInfluence } = interpolate;
    if (bookA === null || bookB === null) return;
    const A = books[bookA];
    const B = books[bookB];
    if (!A || !B) return;

    // Build a field lookup if data is present and the slider is non-zero.
    // The adaptive octree cells aren't on a regular lattice, so we use a
    // brute-force nearest-cell scan. With ~4k cells and N=60 sample points,
    // this is ~250k distance comparisons per slider tick — trivial.
    let fieldLookup: ((x: number, y: number, z: number) => [number, number, number]) | null = null;
    let fieldDeformScale = 0;
    if (fieldInfluence > 0 && metadata?.field) {
      const grid = metadata.field.grid;
      // Hoist coords into typed arrays for tight inner loops.
      const nGrid = grid.length;
      const gx = new Float32Array(nGrid);
      const gy = new Float32Array(nGrid);
      const gz = new Float32Array(nGrid);
      const gvx = new Float32Array(nGrid);
      const gvy = new Float32Array(nGrid);
      const gvz = new Float32Array(nGrid);
      for (let i = 0; i < nGrid; i++) {
        const c = grid[i];
        gx[i] = c.x; gy[i] = c.y; gz[i] = c.z;
        gvx[i] = c.vx; gvy[i] = c.vy; gvz[i] = c.vz;
      }
      fieldLookup = (x: number, y: number, z: number) => {
        let bestI = 0;
        let bestD = Infinity;
        for (let i = 0; i < nGrid; i++) {
          const dx = x - gx[i];
          const dy = y - gy[i];
          const dz = z - gz[i];
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < bestD) { bestD = d2; bestI = i; }
        }
        return [gvx[bestI], gvy[bestI], gvz[bestI]];
      };
      // Pick a deformation scale so that fieldInfluence=1 produces a
      // visible bend (~3 units) given the median grid magnitude.
      const norms = grid.map((c) => Math.hypot(c.vx, c.vy, c.vz)).sort((a, b) => a - b);
      const med = norms[Math.floor(norms.length / 2)] || 1;
      fieldDeformScale = 4 / med;
    }

    // Sample 60 evenly spaced positions along [0, 1]; for each, look
    // up the closest-position passage in each book and lerp.
    const N = 60;
    const positions: number[] = [];
    const samples: { pos: number; pt: THREE.Vector3 }[] = [];
    const findClosest = (book: Book, t: number): Passage => {
      let bestI = 0, bestD = Infinity;
      for (let i = 0; i < book.passages.length; i++) {
        const d = Math.abs(book.passages[i].pos - t);
        if (d < bestD) { bestD = d; bestI = i; }
      }
      return book.passages[bestI];
    };
    for (let k = 0; k < N; k++) {
      const t = k / (N - 1);
      const pa = findClosest(A, t);
      const pb = findClosest(B, t);
      const [ax, ay, az] = passageCoord(pa, 'default');
      const [bx, by, bz] = passageCoord(pb, 'default');
      let x = ax * (1 - alpha) + bx * alpha;
      let y = ay * (1 - alpha) + by * alpha;
      let z = az * (1 - alpha) + bz * alpha;
      if (fieldLookup) {
        const [vx, vy, vz] = fieldLookup(x, y, z);
        x += vx * fieldInfluence * fieldDeformScale;
        y += vy * fieldInfluence * fieldDeformScale;
        z += vz * fieldInfluence * fieldDeformScale;
      }
      positions.push(x, y, z);
      samples.push({ pos: t, pt: new THREE.Vector3(x, y, z) });
    }

    // Draw both endpoint trajectories at low opacity
    const drawBook = (book: Book, hex: string) => {
      const pts: number[] = [];
      for (const p of book.passages) {
        const [x, y, z] = passageCoord(p, 'default');
        pts.push(x, y, z);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
      const m = new THREE.LineBasicMaterial({ color: new THREE.Color(hex), transparent: true, opacity: 0.35 });
      s.interpGroup.add(new THREE.Line(g, m));
    };
    drawBook(A, BOOK_COLOURS[bookA % BOOK_COLOURS.length]);
    drawBook(B, BOOK_COLOURS[bookB % BOOK_COLOURS.length]);

    // Interpolated polyline in burnt orange
    const interpGeom = new THREE.BufferGeometry();
    interpGeom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const interpMat = new THREE.LineBasicMaterial({ color: 0xC0392B, transparent: true, opacity: 0.9 });
    s.interpGroup.add(new THREE.Line(interpGeom, interpMat));

    // Markers at start, mid, end of the interpolated path
    const markerColour = new THREE.Color('#3d3328');
    for (const idx of [0, Math.floor(N / 2), N - 1]) {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.18, 12, 12),
        new THREE.MeshBasicMaterial({ color: markerColour })
      );
      sphere.position.copy(samples[idx].pt);
      s.interpGroup.add(sphere);
    }
  }, [books, viewMode, interpolate, metadata]);

  // Hit testing on mouse move
  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const s = stateRef.current;
      if (!s) return;
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const mx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const my = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      s.raycaster.setFromCamera(new THREE.Vector2(mx, my), s.camera);

      // Find nearest point within threshold
      let closest: HoveredPassage | null = null;
      let minDist = 0.4; // world-space threshold

      for (const entry of s.allPositions) {
        const dist = s.raycaster.ray.distanceToPoint(entry.position);
        if (dist < minDist) {
          minDist = dist;
          closest = {
            bookIndex: entry.bookIndex,
            passageIndex: entry.passageIndex,
            screenX: e.clientX,
            screenY: e.clientY,
          };
        }
      }
      onHover(closest);
    },
    [onHover]
  );

  const handleMouseLeave = useCallback(() => onHover(null), [onHover]);

  // Double-click to reset view
  const handleDoubleClick = useCallback(() => {
    const s = stateRef.current;
    if (!s || booksRef.current.length === 0) return;
    const center = centerOfData(booksRef.current, viewMode);
    s.controls.target.copy(center);
    s.camera.position.set(center.x + 12, center.y + 8, center.z + 12);
    s.controls.update();
  }, [viewMode]);

  return (
    <div
      ref={containerRef}
      className="flex-1 h-screen overflow-hidden relative cursor-grab active:cursor-grabbing"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onDoubleClick={handleDoubleClick}
    />
  );
}
