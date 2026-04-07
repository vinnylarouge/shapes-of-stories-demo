export interface PassageNeighbour {
  b: number;  // book index
  p: number;  // passage index
  d: number;  // Euclidean distance in latent space
}

export interface Passage {
  x: number;
  y: number;
  x3d: number;
  y3d: number;
  z3d: number;
  pos: number;
  text: string;
  dialogue: number;
  entropy: number;
  sent_len: number;
  // E01 — cross-book nearest passages.
  neighbours?: PassageNeighbour[];
  // E03 — motif code from k-means.
  code?: number;
  // E04 — surprise residual against the kernel vector field.
  surprise?: number;
  // E05 — projection onto top-3 averaged canonical axes.
  canonical?: [number, number, number];
}

export interface PersistencePair {
  d: 0 | 1;
  birth: number;
  death: number;
}

export interface Book {
  title: string;
  full_length: number;
  archetype: number;
  passages: Passage[];
  // E02 — per-book persistence signature and shape archetype.
  persistence?: PersistencePair[];
  shape_archetype?: number;
  // E03 — per-book motif histogram (length = metadata.vq_k).
  code_histogram?: number[];
}

export interface FieldGridPoint {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  // Optional half-cell-size when the field is an adaptive octree
  // (E04 Barnes-Hut sampling). Smaller h = denser refinement at this
  // location.
  h?: number;
}

export interface Metadata {
  model: string;
  layer: number;
  window: number;
  ae_k: number;
  n_books: number;
  n_passages: number;
  // E01
  neighbours_k?: number;
  // E02
  persistence_top?: number;
  shape_archetypes?: number;
  // E03
  vq_k?: number;
  vq_centroids?: number[][];
  // E04
  field?: { grid: FieldGridPoint[]; bandwidth: number; grid_n: number };
  // E05
  canonical_axes?: {
    samples_per_book: number;
    n_pairs: number;
    mean_canonical_corrs: number[];
  };
}

export interface StoryData {
  metadata: Metadata;
  books: Book[];
}

export type ColourMode =
  | 'book'
  | 'dialogue'
  | 'entropy'
  | 'sent_len'
  | 'position'
  | 'motif'      // E03
  | 'surprise'   // E04
  | 'shape';     // E02

export type ViewMode = 'default' | 'canonical' | 'interpolate';

export interface InterpolateState {
  bookA: number | null;
  bookB: number | null;
  alpha: number;
  // E04 ↔ E06 coupling: how strongly the interpolated trajectory is
  // bent by the local vector field (0 = pure lerp, 1 = strongly bent).
  fieldInfluence: number;
}

export interface HoveredPassage {
  bookIndex: number;
  passageIndex: number;
  screenX: number;
  screenY: number;
}
