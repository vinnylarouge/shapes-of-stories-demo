import { useRef, useEffect, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Book, ColourMode, HoveredPassage } from '../utils/types';
import { BOOK_COLOURS, getFeatureColour } from '../utils/colours';

interface Props {
  books: Book[];
  activeBooks: Set<number>;
  colourMode: ColourMode;
  hovered: HoveredPassage | null;
  onHover: (h: HoveredPassage | null) => void;
  playback: { bookIndex: number; passageIndex: number } | null;
  hoveredBookIndex: number | null;
}

function hexToThreeColor(hex: string): THREE.Color {
  return new THREE.Color(hex);
}

function centerOfData(books: Book[]) {
  let sx = 0, sy = 0, sz = 0, n = 0;
  for (const b of books) {
    for (const p of b.passages) {
      sx += p.x3d; sy += p.y3d; sz += p.z3d; n++;
    }
  }
  return new THREE.Vector3(sx / n, sy / n, sz / n);
}

export function Canvas({ books, activeBooks, colourMode, hovered, onHover, playback, hoveredBookIndex }: Props) {
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
    scene.add(pointsGroup);
    scene.add(linesGroup);
    scene.add(highlightGroup);

    stateRef.current = {
      scene, camera, renderer, controls,
      raycaster: new THREE.Raycaster(),
      mouse: new THREE.Vector2(),
      pointsGroup, linesGroup, highlightGroup,
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

  // Center camera on data once loaded
  useEffect(() => {
    if (!stateRef.current || books.length === 0) return;
    const center = centerOfData(books);
    stateRef.current.controls.target.copy(center);
    stateRef.current.camera.position.set(center.x + 12, center.y + 8, center.z + 12);
    stateRef.current.controls.update();
  }, [books.length === 0]); // eslint-disable-line react-hooks/exhaustive-deps

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
        const pos = new THREE.Vector3(p.x3d, p.y3d, p.z3d);
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
          const fc = getFeatureColour(colourMode, p);
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
        linePositions.push(p.x3d, p.y3d, p.z3d);
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
        const startGeom = new THREE.SphereGeometry(0.15, 12, 12);
        const startMat = new THREE.MeshBasicMaterial({ color: bookColour });
        const startMesh = new THREE.Mesh(startGeom, startMat);
        startMesh.position.set(p0.x3d, p0.y3d, p0.z3d);
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
        const endGeom = new THREE.BoxGeometry(0.2, 0.2, 0.2);
        const endMat = new THREE.MeshBasicMaterial({ color: bookColour });
        const endMesh = new THREE.Mesh(endGeom, endMat);
        endMesh.position.set(pN.x3d, pN.y3d, pN.z3d);
        s.linesGroup.add(endMesh);
      }
    }
  }, [books, activeBooks, colourMode, hoveredBookIndex]);

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
        mesh.position.set(p.x3d, p.y3d, p.z3d);
        s.highlightGroup.add(mesh);
      }

      // Glow ring on current
      const curr = book.passages[playback.passageIndex];
      const ringGeom = new THREE.RingGeometry(0.25, 0.32, 24);
      const ringMat = new THREE.MeshBasicMaterial({
        color: bookColour,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(ringGeom, ringMat);
      ring.position.set(curr.x3d, curr.y3d, curr.z3d);
      ring.lookAt(s.camera.position);
      s.highlightGroup.add(ring);
    }

    if (hovered) {
      const p = books[hovered.bookIndex].passages[hovered.passageIndex];
      const bookColour = hexToThreeColor(BOOK_COLOURS[hovered.bookIndex % BOOK_COLOURS.length]);
      const ringGeom = new THREE.RingGeometry(0.2, 0.26, 24);
      const ringMat = new THREE.MeshBasicMaterial({
        color: bookColour,
        side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(ringGeom, ringMat);
      ring.position.set(p.x3d, p.y3d, p.z3d);
      ring.lookAt(s.camera.position);
      s.highlightGroup.add(ring);
    }
  }, [books, playback, hovered]);

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
    const center = centerOfData(booksRef.current);
    s.controls.target.copy(center);
    s.camera.position.set(center.x + 12, center.y + 8, center.z + 12);
    s.controls.update();
  }, []);

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
